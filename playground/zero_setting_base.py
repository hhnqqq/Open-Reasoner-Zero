from typing import List

from jinja2 import Template
from orz.ppo import PromptDataset


class CustomDataset(PromptDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_dialogue(self, dialogue: List):
        system_prompt = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. \
The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. """

        prompt_instruction_template_jinja = """\
Please reason step by step, and put your final answer within \\boxed{} for closed-end problem (including multiple choice problem).
If the problem requires multiple answers to be answered, place all final answers in one \\boxed{} environment, separated by commas
If the problem is in Chinese, provide your reasoning and answer in Chinese. Otherwise, use English.
Problem: {{prompt}}
"""

        instruction_template = Template(prompt_instruction_template_jinja)
        instruction = instruction_template.render(prompt=dialogue["problem"])
        system_content = {"content": system_prompt, "role":"system"}
        input_content = {"content": instruction, "role": "user"}
        messages = [system_content,input_content]
        prompt = self.tokenizer.apply_chat_template(messages,
                                                    tokenize=False,
                                                    add_generation_prompt=True)
        # A lot of solutions are None.
        extra = {"answer": dialogue["answer"], 
                 "solution": dialogue["solution"] if "solution" in dialogue.keys() else " ",
                 "file_name": dialogue["file_name"]}

        return prompt, extra


class EvalCustomDataset(PromptDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_dialogue(self, dialogue: dict):
        system_prompt = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. \
The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. """

        prompt_instruction_template_jinja = """\
Please reason step by step, and put your final answer within \\boxed{} for closed-end problem (including multiple choice problem).
If the problem requires multiple answers to be answered, place all final answers in one \\boxed{} environment, separated by commas
If the problem is in Chinese, provide your reasoning and answer in Chinese. Otherwise, use English.
Problem: {{prompt}}
"""
        assert isinstance(dialogue, dict), "dialogue must be a dict"
        assert "problem" in dialogue, "dialogue must contain problem"
        assert "answer" in dialogue, "dialogue must contain answer"
        assert "file_name" in dialogue, "dialogue must contain file_name"

        instruction_template = Template(prompt_instruction_template_jinja)
        instruction = instruction_template.render(prompt=dialogue["problem"])
        system_content = {"content": system_prompt, "role":"system"}
        input_content = {"content": instruction, "role": "user"}
        messages = [system_content,input_content]
        prompt = self.tokenizer.apply_chat_template(messages,
                                                    tokenize=False,
                                                    add_generation_prompt=True)

        extra = {"answer": dialogue["answer"], "file_name": dialogue["file_name"]}

        return prompt, extra
