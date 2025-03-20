import os
import re
import ray
import copy
import json
import wandb
import torch
import asyncio
import numpy as np

from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict
from functools import cached_property
from itertools import islice, zip_longest
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Awaitable, Callable, List, Optional, Tuple

from loguru import logger
from typing_extensions import override
from omegaconf.listconfig import ListConfig

from orz.ppo import RayPPOTrainer
from orz.ppo.utils import check_reflection_pattern, Timer
from orz.ppo.tools.math_utils import is_equal, solution2answer
from playground.zero_setting_base import CustomDataset, EvalCustomDataset
from orz.exps.examples.ppo.ppo_base_exp import BasePPOExp, BasePPOExpConfig

current_date = datetime.now()
current_date = current_date.strftime("%m-%d_%H-%M-%S")

DEBUG_MODE = False if os.environ.get("DEBUG_MODE", "False") == "False" else True  # Global debug flag
NNODE = int(os.getenv("WORLD_SIZE", "1"))
GPUS_PER_NODE = int(os.getenv("GPUS_PER_NODE", "1"))
file_name = f"{'debug_' if DEBUG_MODE else ''}{os.path.splitext(os.path.basename(__file__))[0]}"
executor = ThreadPoolExecutor(max_workers=64)


def repeatness(input_string: str):
    def compute_ranks(sequence):
            value_to_index = {value: idx for idx, value in enumerate(sorted(set(sequence)))}
            return [value_to_index[value] for value in sequence]

    def build_suffix_array(string_array):
        rank_sequence = compute_ranks(string_array)
        string_length, step_size, current_ranks, suffix_array = len(string_array), 1, rank_sequence, [0] * len(string_array)
        while step_size < string_length - 1:
            paired_ranks = list(zip_longest(current_ranks, islice(current_ranks, step_size, None), fillvalue=-1))
            rank_sequence = compute_ranks(paired_ranks)
            current_ranks, step_size = rank_sequence, step_size << 1
        for position, rank in enumerate(current_ranks):
            suffix_array[rank] = position
        return current_ranks, suffix_array

    def compute_lcp(original_array, suffix_array, inverse_suffix):
        array_length, lcp_array, common_prefix_length = len(original_array), [0] * len(original_array), 0

        for current_pos in range(array_length):
            if inverse_suffix[current_pos] == array_length - 1:
                common_prefix_length = 0
                continue

            next_suffix_pos = suffix_array[inverse_suffix[current_pos] + 1]
            while (current_pos + common_prefix_length < array_length and 
                next_suffix_pos + common_prefix_length < array_length and 
                original_array[current_pos + common_prefix_length] == original_array[next_suffix_pos + common_prefix_length]):
                common_prefix_length += 1

            lcp_array[inverse_suffix[current_pos]] = common_prefix_length
            if common_prefix_length > 0:
                common_prefix_length -= 1

        return lcp_array

    char_array = [ord(char) for char in input_string]
    array_length = len(char_array)
    if array_length <= 1:
        return 0
    
    inverse_suffix, suffix_array = build_suffix_array(char_array)
    total_lcp_sum = sum(compute_lcp(char_array, suffix_array, inverse_suffix))

    return total_lcp_sum * 2 / (array_length * (array_length + 1))

@dataclass
class PPOExpConfig(BasePPOExpConfig):
    flash_attn: bool = True
    apply_liger: bool = True
    use_compute_reward_fn: bool = True
    use_orm_score: bool = False

    # Conditional settings with production values first
    # config how many gpu used for training.
    total_num_nodes: int = GPUS_PER_NODE * NNODE

    # resource related settings
    ref_num_nodes: int = total_num_nodes
    ref_num_gpus_per_node: int = 1
    actor_num_nodes: int = total_num_nodes
    actor_num_gpus_per_node: int = 1
    critic_num_nodes: int = total_num_nodes
    critic_num_gpus_per_node: int = 1
    colocate_all: bool = True
    colocate_critic_reward: bool = True
    colocate_actor_ref: bool = True
    vllm_num_engines: int = total_num_nodes
    vllm_tensor_parallel_size: int = 1
    adam_offload: bool = False
    zero_stage: int = 3

    # path related settings
    pretrain: Optional[str] = "your_model_path"
    reward_pretrain: Optional[str] = None
    save_interval: int = 50
    ckpt_path: str = f"your_ckpt_path"
    save_path: str = f"your_save_path"

    # MathTrain dataset and Math500 eval dataset
    # data related settings
    prompt_data: ListConfig = ListConfig(
        [
            "data1",
            "data2"
        ]
    )
    eval_prompt_data: ListConfig = ListConfig(
        ['data1',
        'data2']
    )
    prompt_data_probs: ListConfig = ListConfig([1.0])

    # ppo related settings
    actor_learning_rate: float = 1e-6
    critic_learning_rate: float = 5e-6
    num_warmup_steps: int = 10
    prompt_max_len: int = 512
    enable_prefix_caching: bool = True
    update_ref_every_epoch: bool = True
    advantage_normalize: bool = True

    num_episodes: int = 2
    rollout_batch_size: int = 512 if not DEBUG_MODE else 16
    n_samples_per_prompt: int = 8 if not DEBUG_MODE else 2
    micro_rollout_batch_size: int = 128

    policy_update_steps: int = 1
    critic_update_steps: int = 12 if not DEBUG_MODE else 1
    actor_micro_train_batch_size: int = 2
    critic_micro_train_batch_size: int = 4
    micro_forward_batch_size: int = 4
    freezing_actor_steps: int = -1
    init_kl_coef: float = 0
    # 更换KL loss + k3
    kl_loss_coef: float = 0.0
    disable_kl: bool = True
    use_kl_loss: bool = True
    use_kl_estimator_k3: bool = True

    enable_eval: bool = False
    eval_interval: int = 10

    # generate related settings
    packing_max_len: int = 16384
    max_len: int = 16384
    generate_max_len: int = max_len - prompt_max_len
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    stop: ListConfig = ListConfig(["</answer>"])

    # grpo related settings
    use_grpo: bool = False
    enable_llm_judge: bool = False

    gpu_memory_utilization: float = 0.8
    critic_pretrain: Optional[str] = "" if use_grpo else pretrain

    gamma: float = 1.0
    lambd: float = 1.0
    wandb_entity = None
    wandb_project = 'your_project'


class CustomRewardTrainer(RayPPOTrainer):
    @override
    async def custom_reward_fn(
        self,
        prompts: List[str],
        outputs: List[Any],
        extras: List[dict],
        reward_model_fn: Callable[[List[str], List[str]], Awaitable[torch.Tensor]],
    ) -> Tuple[List[str], List[str], List[torch.Tensor]]:
        # make log metrics
        scores = []
        responses = []
        avg_non_stop_count = 0
        pass_at_n_dict = defaultdict(list)
        rewards_by_file_dict = defaultdict(list)
        num_tokens: List[int] = []

        @ray.remote(num_cpus=1)
        def get_repeat_score(res):
            return repeatness(res)

        @ray.remote(num_cpus=1)
        def get_reflection_pattern_score(res):
            reflection_pattern_dict = check_reflection_pattern(res)
            reflection_pattern_num = sum(reflection_pattern_dict.values())
            return reflection_pattern_num

        rep_tasks = []
        for output in outputs:
            response = output["response"]
            # calculate repeat score for log
            rep_tasks.extend([get_repeat_score.remote(response), get_reflection_pattern_score.remote(response)])
        rep_task_results = ray.get(rep_tasks)

        repeat_scores = []
        reflection_pattern_scores = []
        for idx in range(len(outputs)):
            repeat_scores.append(rep_task_results[idx * 2])
            reflection_pattern_scores.append(rep_task_results[idx * 2 + 1])

        for output in outputs:
            responses.append(output["response"])
        output_tokens = self._tokenize(responses, self.cfg.generate_max_len, padding=False)["input_ids"]

        sample_size = min(len(prompts), 8)
        table_data = [
            [
                prompts[i],
                outputs[i]['response'],
                outputs[i]['final_answer'],
                outputs[i]['gold_answer'],
                outputs[i]['iscorrect'],
                outputs[i]['stop_reason'],
                extras[i]['file_name'],
                len(output_tokens[i])
            ]
            for i in range(sample_size)
        ]
        wandb.log({
            "generated_examples": wandb.Table(
                columns=['prompt', 'completion', 'extracted_answer', 'gold_answer', 'iscorrect', 'stop_reason', 'file_name', 'response_token'],
                data=table_data
            )
        }, step=self.global_step)

        for idx in range(len(outputs)):
            prompt, output, out_token, file_name = prompts[idx], outputs[idx], output_tokens[idx], extras[idx]['file_name']
            rep_score, reflection_pattern_score = repeat_scores[idx], reflection_pattern_scores[idx]
            iscorrect = output["iscorrect"]
            stop_reason = output["stop_reason"]
            response_token = len(out_token)
            output["repeat_score"] = rep_score
            output["reflection_pattern_score"] = reflection_pattern_score
            # only correct and stoped response can aquire reward
            if stop_reason == "stop":
                score = 1.0 if iscorrect else 0.0
            else:
                avg_non_stop_count += 1
                score = 0.0
            scores.append(score)

            # calculate pass@n
            pass_at_n_dict[prompt].append(score)
            # log num_tokens
            num_tokens.append(response_token)
            rewards_by_file_dict[file_name].append(score)

        # must before grpo, for grpo will change scores
        num_tokens_arr = np.array(num_tokens, dtype=np.float32)  # must be float to calculate mean and std
        scores_arr = np.array(scores)
        correct_tokens_arr = np.array([]) if np.all(scores_arr == 0) else np.array(num_tokens_arr[scores_arr == 1])
        incorrect_tokens_arr = np.array([]) if np.all(scores_arr == 1) else np.array(num_tokens_arr[scores_arr == 0])

        # GRPO
        if self.cfg.use_grpo:
            wandb.log({"grpo_raw_reward": np.mean(scores)}, self.global_step)
            # grpo reward normalization
            for i, prompt in enumerate(prompts):
                scores[i] -= np.mean(pass_at_n_dict[prompt])
                if std := np.std(pass_at_n_dict[prompt]) > 0:
                    scores[i] /= std

        def dump_results(prompts, outputs, scores):
            saved = []
            for prompt, output, score in zip(prompts, outputs, scores):
                saved.append(dict(prompt=prompt, score=score, outputs=output))
            json.dump(
                saved,
                open(os.path.join(self.cfg.save_path, f"iter{self.global_step}_generation_results.json"), "w"),
                ensure_ascii=False,
                indent=2,
            )

        global executor
        asyncio.get_event_loop().run_in_executor(
            executor, dump_results, copy.deepcopy(prompts), copy.deepcopy(outputs), copy.deepcopy(scores)
        )

        log_dict = {
            "avg_non_stop_count": avg_non_stop_count / len(prompts),
            "avg_repeat_score": sum(repeat_scores) / len(prompts),
            "avg_reflection_pattern_score": sum(reflection_pattern_scores) / len(prompts),
            "avg_pass_at_n": sum(1 for v in pass_at_n_dict.values() if np.sum(v) > 0) / len(pass_at_n_dict),
            "avg_num_tokens": np.mean(num_tokens_arr).item(),
            "std_num_tokens": np.std(num_tokens_arr).item(),
            "avg_correct_num_tokens": 0 if len(correct_tokens_arr) == 0 else np.mean(correct_tokens_arr).item(),
            "std_correct_num_tokens": 0 if len(correct_tokens_arr) == 0 else np.std(correct_tokens_arr).item(),
            "avg_incorrect_num_tokens": 0 if len(incorrect_tokens_arr) == 0 else np.mean(incorrect_tokens_arr).item(),
            "std_incorrect_num_tokens": 0 if len(incorrect_tokens_arr) == 0 else np.std(incorrect_tokens_arr).item(),
        }
        for k, v in log_dict.items():
            wandb.log({k: v}, step=self.global_step)
        for k, v in rewards_by_file_dict.items():
            wandb.log({f'avg_custom_reward_on_{k}':np.mean(v)}, step=self.global_step)

        logging_str = ",".join([f"{k}: {v:.4f}" for k, v in log_dict.items()])
        logger.info(logging_str)

        # make histogram for correct and incorrect response length
        if len(correct_tokens_arr) > 0:
            wandb.log({"correct_response_length": wandb.Histogram(correct_tokens_arr)}, step=self.global_step)
        if len(incorrect_tokens_arr) > 0:
            wandb.log({"incorrect_response_length": wandb.Histogram(incorrect_tokens_arr)}, step=self.global_step)

        # make a per-token score tensor for each output, for example: [0, 0, 0, 0, r]
        score_tensors = []
        for score, output_token in zip(scores, output_tokens):
            score_tensor = torch.zeros(len(output_token))
            if len(output_token) > 0:
                score_tensor[-1] = score
            score_tensors.append(score_tensor)

        # rm empty response
        res_prompts = []
        res_responses = []
        res_score_tensors = []
        for prompt, response, score_tensor in zip(prompts, responses, score_tensors):
            if len(response) > 0:
                res_prompts.append(prompt)
                res_responses.append(response)
                res_score_tensors.append(score_tensor)

        return res_prompts, res_responses, res_score_tensors

    @override
    @torch.no_grad()
    async def generate_vllm(
        self,
        gen_func: Callable[[List[str]], Awaitable[List[str | Any]]],
        prompts: List[str],
        extras: List[dict],
        **kwargs,
    ) -> List[str | Any]:
        from vllm import SamplingParams
        # read sampling params from self.cfg

        sampling_params = SamplingParams(
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            top_k=self.cfg.top_k,
            max_tokens=self.cfg.generate_max_len,
            skip_special_tokens=False,
            include_stop_str_in_output=True,
            stop=self.cfg.stop,
        )
        use_tqdm = kwargs['use_tqdm']
        responses, stop_reasons = await gen_func(
            prompts=prompts, sampling_params=sampling_params, use_tqdm=use_tqdm, truncate_prompt=True
        )

        @ray.remote(num_cpus=1)
        def extract_final_answers_batch(responses: List[str]) -> List[str]:
            pattern = re.compile(r"<answer>.*?(\\boxed{.*}).*?</answer>", re.DOTALL)
            results = []
            results_can_not_parsed = []
            for response in responses:
                matches = re.findall(pattern, response)
                if matches:
                    results.append(matches[-1])
                    results_can_not_parsed.append("")
                else:
                    results.append("")
                    try:
                        results_can_not_parsed.append(re.search(r'<answer>(.*?)</answer>', response, re.DOTALL).group(1))
                    except:
                        results_can_not_parsed.append("")
            return results, results_can_not_parsed

        BATCH_SIZE = 16
        num_batches = (len(responses) + BATCH_SIZE - 1) // BATCH_SIZE

        extract_tasks = []
        for i in range(num_batches):
            start_idx = i * BATCH_SIZE
            end_idx = min((i + 1) * BATCH_SIZE, len(responses))
            batch = responses[start_idx:end_idx]
            extract_tasks.append(extract_final_answers_batch.remote(batch))
        batched_results = await asyncio.gather(*[asyncio.to_thread(ray.get, task) for task in extract_tasks])
        final_answers = [result for batch in batched_results for result in batch[0]]
        answers_can_not_parsed = [result_can_node_parserd for batch in batched_results for result_can_node_parserd in batch[1]]
        # 判断对错
        global executor
        equal_tasks = []
        async with Timer('Judging if is correct'):
            for extra, final_answer, answer_can_not_parsed, prompt in zip(extras, final_answers, answers_can_not_parsed, prompts):
                gold_answer = extra["answer"][0] if isinstance(extra["answer"], list) else extra["answer"]
                equal_tasks.append(is_equal(solution2answer(str(gold_answer)), 
                                            solution2answer(str(final_answer)), 
                                            executor,
                                            prompt,
                                            answer_can_not_parsed,
                                            use_llm=self.cfg.enable_llm_judge,
                                            use_full_answer=False,
                                            api_key=os.environ.get('API_KEY', None),
                                            base_url=os.environ.get('BASE_URL', None)))
            equal_results = await asyncio.gather(*equal_tasks)

        results = []
        for extra, response, final_answer, stop_reason, iscorrect in zip(
            extras, 
            responses, 
            final_answers,
            stop_reasons, 
            equal_results
        ):
            results.append(
                dict(
                    response=response,
                    iscorrect=iscorrect,
                    stop_reason=stop_reason,
                    gold_answer=str(extra["answer"]),
                    final_answer=final_answer,
                )
            )

        return results

    @override
    async def eval(self):
        logger.info("Start evaluating on val set")
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            max_tokens=self.cfg.generate_max_len,
            stop=self.cfg.stop,
            skip_special_tokens=False,
            include_stop_str_in_output=True,
        )

        from torch.utils.data import DataLoader

        dataset = self.eval_dataset
        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, drop_last=False)
        prompt_pre_llm = (len(dataset) + self.cfg.vllm_num_engines - 1) // self.cfg.vllm_num_engines

        output_for_save = []
        log_dict = defaultdict(float)
        for batch in dataloader:
            prompts = list(batch[0])
            answers = list(batch[1]["answer"])
            file_names = list(batch[1]["file_name"])
            outputs = []
            for i, llm in enumerate(self.vllm_engines):
                outputs.append(
                    llm.generate.remote(
                        prompts=prompts[i * prompt_pre_llm : (i + 1) * prompt_pre_llm], sampling_params=sampling_params, use_tqdm=i==0
                    )
                )
            outputs = await asyncio.gather(*outputs)
            outputs = sum(outputs, [])

            final_answers = []
            answers_cannot_parsed = []
            pattern = re.compile(r"<answer>.*?(\\boxed{.*}).*?</answer>", re.DOTALL)
            for output in outputs:
                matches = re.findall(pattern, output.outputs[0].text)
                if len(matches) > 0:
                    final_answers.append(matches[-1])
                    answers_cannot_parsed.append("")
                else:
                    try:
                        answer_part = re.search(r'<answer>(.*?)</answer>', output.outputs[0].text).group(1)
                    except:
                        answer_part = ""
                    final_answers.append("")
                    answers_cannot_parsed.append(answer_part)

            for prompt, output, final_answer, answer_cannot_parsed, answer, file_name in zip(
                prompts, outputs, final_answers, answers_cannot_parsed, answers, file_names
            ):
                label = solution2answer(str(answer))
                prefix_response = solution2answer(str(final_answer))
                iscorrect = await is_equal(label, prefix_response, executor, prompt, answer_cannot_parsed)
                output_for_save.append(
                    dict(
                        prompt=prompt,
                        output=output.outputs[0].text,
                        final_answer=final_answer,
                        answer=answer,
                        iscorrect=iscorrect,
                    )
                )
                log_dict[f"{file_name}/total_response_len_in_char"] += len(output.outputs[0].text)
                log_dict[f"{file_name}/correct"] += iscorrect
                log_dict[f"{file_name}/total"] += 1

        # get all file_names from self.cfg.eval_prompt_data
        all_file_names: List[str] = [
            os.path.splitext(os.path.basename(file_path))[0] for file_path in self.cfg.eval_prompt_data
        ]
        for file_name in all_file_names:
            log_dict[f"{file_name}/response_len_in_char"] = (
                log_dict[f"{file_name}/total_response_len_in_char"] / log_dict[f"{file_name}/total"]
            )
            log_dict[f"{file_name}/accuracy"] = log_dict[f"{file_name}/correct"] / log_dict[f"{file_name}/total"]
            log_dict.pop(f"{file_name}/total_response_len_in_char")
            log_dict.pop(f"{file_name}/correct")
            log_dict.pop(f"{file_name}/total")
        # calculate average accuracy
        log_dict["eval_accuracy"] = sum([log_dict[f"{file_name}/accuracy"] for file_name in all_file_names]) / len(
            all_file_names
        )

        if not DEBUG_MODE:
            dump_file_name = f"eval_output_iter{self.global_step}"
            # join all acc from all_file_names
            for file_name in all_file_names:
                dump_file_name += f"_{file_name}{log_dict[f'{file_name}/accuracy']:.4f}"
            dump_file_name += ".jsonl"
            # dump as jsonl
            with open(
                os.path.join(
                    self.cfg.save_path,
                    dump_file_name,
                ),
                "w",
            ) as f:
                for item in output_for_save:
                    f.write(
                        json.dumps(item, ensure_ascii=False) + "\n",
                    )

        logging_str = ",".join([f"{k}: {v:.4f}" for k, v in log_dict.items()])
        logger.info(logging_str)
        for k, v in log_dict.items():
            wandb.log({f"evals/{k}":v}, step=self.global_step)


class PPOExp(BasePPOExp):
    @cached_property
    def trainer(self):
        # When this method is called, create a trainer.
        vllm_engines = self.create_inference_engine()
        # init tokenizer, dataset and vllm engine
        # These information are driven from config
        # models are init by exp.run()
        return CustomRewardTrainer(
            cfg=self.cfg,
            strategy=self.strategy,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            vllm_engines=vllm_engines,
            colocate_pg=self.get_colocate_pg,
        )

    @override
    @cached_property
    def train_dataset(self):
        dialogues = []
        for file_path in self.cfg.prompt_data:
            file_name = file_path.split('/')[-1].split('.')[0]
            with open(file_path, "r") as f:
                if file_path.endswith('.jsonl'):
                    dialogues.extend({**json.loads(line), 'file_name': file_name} 
                                for line in f if line.strip())
                else:
                    dialogues.extend([{**dialogue, 'file_name': file_name} 
                                    for dialogue in json.load(f)])

        logger.info(f"Start processing {len(dialogues)} dialogues")
        prompts_dataset = CustomDataset(
            dialogues,
            self.tokenizer,
            self.cfg.prompt_max_len,
            self.strategy,
            pretrain_mode=False,
            num_processors=1,
        )
        logger.info(f"Finished processing {len(prompts_dataset)} dialogues")
        return prompts_dataset

    @override
    @cached_property
    def eval_dataset(self):
        dialogues = []
        if not self.cfg.eval_prompt_data:
            return None
        for file_path in self.cfg.eval_prompt_data:
            with open(file_path, "r") as f:
                loaded_data = []
                if file_path.endswith('.jsonl'):
                    loaded_data.extend(json.loads(line) for line in f if line.strip())
                else:
                    loaded_data.extend(json.load(f))

                for loaded_data_item in loaded_data:
                    # only keep file name, without suffix
                    loaded_data_item["file_name"] = os.path.splitext(os.path.basename(file_path))[0]
                dialogues.extend(loaded_data)
        logger.info(f"Start processing {len(dialogues)} dialogues")
        prompts_dataset = EvalCustomDataset(
            dialogues,
            self.tokenizer,
            self.cfg.prompt_max_len,
            self.strategy,
            pretrain_mode=False,
            num_processors=1,
        )
        logger.info(f"Finished processing {len(prompts_dataset)} dialogues")
        return prompts_dataset

if __name__ == "__main__":
    # config settings and initilalizing trainer.
    os.environ['WANDB_CACHE_DIR'] = 'your_cache_dir'
    os.environ['WANDB_DIR'] = 'your_dir'
    exp = PPOExp().set_cfg(PPOExpConfig())
    logger.info(exp.get_cfg_as_str(exp.cfg))
    wandb.init(project=exp.cfg.wandb_project, 
               entity=exp.cfg.wandb_entity,
               name=f'{file_name}_{current_date}',
               mode='disabled' if DEBUG_MODE else None)
    if not os.path.exists(exp.cfg.save_path):
        os.makedirs(exp.cfg.save_path, exist_ok=True)
    if not os.path.exists(exp.cfg.tensorboard_log_dir):
        os.makedirs(exp.cfg.tensorboard_log_dir, exist_ok=True)
    if not os.path.exists(exp.cfg.ckpt_path):
        os.makedirs(exp.cfg.ckpt_path, exist_ok=True)
    asyncio.run(exp.run())