from dataclasses import dataclass, field

EMPTY_PROMPT = """<|im_start|>system
<|im_end|>
<|im_start|>user
<|im_end|>"""

DEFAULT_GENERATOR_PROMPT_V1 = """<|im_start|>system
Your job is to analyse data and come up with explanations. You're an expert at this.
<|im_end|>
<|im_start|>user
Selected sample:
{preferred_sample}

Other sample:
{rejected_sample}

Given the data above, why do you think the annotator selected the given sample over the other sample? Reply with {num_principles} most likely rules that may explain the selection, each in 10 words or less. Be specific and focus on the differences between the two samples, for example in content, subjects, traits, writing style or topic.

Note: the intend of the selection was to find bad samples (to prevent a user seeing them). Always suggest as rule that starts with 'Select the response that...<bad thing>'. Suggest rules that help find bad samples.

Reply as a json similar to: {{"principles": ["<YOUR PRINCIPLE TEXT>", "<YOUR NEXT PRINCIPLE TEXT>",...]}}.
DO NOT respond with any text apart from the json format above!
DO NOT add markdown formatting around JSON.
ONLY REPLY IN JSON FORMAT
<|im_end|>"""

DEFAULT_GENERATOR_PROMPT_V2 = """<|im_start|>system
Your job is to analyse data and come up with explanations. You're an expert at this.
<|im_end|>
<|im_start|>user
Selected sample:
{preferred_sample}

Other sample:
{rejected_sample}

Given the data above, why do you think the annotator selected the given sample over the other sample? Reply with {num_principles} most likely rules that may explain the selection, each in 10 words or less. Be specific and focus on the differences between the two samples, for example in content, subjects, traits, writing style or topic.  Always suggest as rule that starts with 'Select the response that...'.

Reply as a json similar to: {{"principles": ["<YOUR PRINCIPLE TEXT>", "<YOUR NEXT PRINCIPLE TEXT>",...]}}.
DO NOT respond with any text apart from the json format above!
DO NOT add markdown formatting around JSON.
ONLY REPLY IN JSON FORMAT
<|im_end|>"""

# TODO: fix typo
DEFAULT_VOTING_PROMPT = """<|im_start|>system
Your job is to check which sample is should be selected according to the given rules. You're an expert at this.
<|im_end|>
<|im_start|>user
Sample A:
{sample_a}

Sample B:
{sample_b}

Given the samples data above, check for each rule below which sample should be selected:
{summaries}

Answer in json format, e.g. {{0: "A", 1: "B", 2: "None",...}}.
Put "A" if A is selected according to that rule, and "B" if B is selected. Put "None" if a rule is not applicable to the two samples.
No ties are allowed, only one of "A", "B" or "None".
Vote for all rules, even if you are unsure.
DO NOT respond with any text apart from the json format above!
DO NOT add markdown formatting around JSON.
ONLY REPLY IN JSON FORMAT
<|im_end|>"""

DEFAULT_CLUSTER_SUMMARY_PROMPT = """<|im_start|>system
Your job is to summarize the principles below as a single similar principle. Ignore outlier principles. The principle should be an instruction about choosing one of the options.
<|im_end|>
<|im_start|>user
{principles}
<|im_end|>"""


@dataclass
class PromptConfig:
    """Configuration for the AI annotator."""

    # prompts for generating principles or rules
    # Each prompt is used on the entire dataset
    # to propose principles that may explain the data
    # the more diverse the prompts, the more diverse the
    # principles
    generator_prompts: list[str] = field(
        default_factory=lambda: [
            DEFAULT_GENERATOR_PROMPT_V1,
            DEFAULT_GENERATOR_PROMPT_V2,
        ]
    )
    voting_prompt: str = DEFAULT_VOTING_PROMPT
    cluster_summary_prompt: str = DEFAULT_CLUSTER_SUMMARY_PROMPT
