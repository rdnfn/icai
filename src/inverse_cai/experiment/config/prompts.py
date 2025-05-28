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

DEFAULT_PROMPT_GENERATOR_PROMPT_V1 = """<|im_start|>system
Your job is to analyse data and come up with explanations. You're an expert at this.
<|im_end|>
<|im_start|>user
Selected sample:
{preferred_sample}

Other sample:
{rejected_sample}

Given the data above, what features of the instruction were important to why one sample was selected over the other? Reply with {num_principles} most likely instruction features that may contribute to the selection, each in 10 words or less. Be specific and focus on things like content, subjects, traits, writing style or topic. Always suggest a instruction feature that starts with 'Does the instruction...'.

Reply as a json similar to: {{"features": ["<YOUR INSTRUCTION FEATURE TEXT>", "<YOUR NEXT INSTRUCTION FEATURE TEXT>",...]}}.
DO NOT respond with any text apart from the json format above!
DO NOT add markdown formatting around JSON.
ONLY REPLY IN JSON FORMAT
<|im_end|>"""

DEFAULT_PROMPT_GENERATOR_PROMPT_V2 = """<|im_start|>system
Your job is to analyse data and come up with explanations. You're an expert at this.
<|im_end|>
<|im_start|>user
User prompt:
{prompt}

What features of the above user prompt would be important to judge the quality of possible responses? Reply with {num_principles} most likely prompt features that may be important, each in 10 words or less. Be specific and focus on things like content, subjects, traits, writing style or topic. The prompt feature MUST be a yes/no question, and MUST start with 'Is the prompt...'.

Example prompt features:
Is the prompt a creative writing task?
Is the prompt written in an annoyed tone?

Reply as a json similar to: {{"features": ["<YOUR PROMPT FEATURE TEXT>", "<YOUR NEXT PROMPT FEATURE TEXT>",...]}}.
DO NOT respond with any text apart from the json format above!
DO NOT add markdown formatting around JSON.
ONLY REPLY IN JSON FORMAT
<|im_end|>"""

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

Answer in json format, e.g. {{0: "A", 1: "B", 2: "None", 3: "Both",...}}.
Put "A" if A is selected according to that rule, and "B" if B is selected. Put "None" if a rule is not applicable to the two samples.
Put "Both" if both A and B should be selected, and the rule is categorical so it is impossible to select only one.
Otherwise, no ties are allowed, only one of "A", "B" or "None".
Vote for all rules, even if you are unsure.
DO NOT respond with any text apart from the json format above!
DO NOT add markdown formatting around JSON.
ONLY REPLY IN JSON FORMAT
<|im_end|>"""

DEFAULT_PROMPT_VOTING_PROMPT = """<|im_start|>system
Your job is to evaluate whether rules are true or false for the given sample. You're an expert at this.
<|im_end|>
<|im_start|>user
Prompt:
{prompt}

Given the prompt above, check whether each rule below is true or false
{summaries}

Answer in json format, e.g. {{0: true, 1: false,...}}.
Put true if the rule is true, and false if the rule is false.
Vote for all rules, even if you are unsure.
DO NOT respond with any text apart from the json format above!
DO NOT add markdown formatting around JSON.
ONLY REPLY IN JSON FORMAT
<|im_end|>"""

DEFAULT_CLUSTER_SUMMARY_PROMPT = """<|im_start|>system
Your job is to summarize the principles below as a single similar principle. Ignore outlier principles. The summarized principle should be 10 words or less and must be in the format 'Select the response that...'.
<|im_end|>
<|im_start|>user
{principles}
<|im_end|>"""

DEFAULT_PROMPT_CLUSTER_SUMMARY_PROMPT = """<|im_start|>system
Your job is to summarize the prompt features below as a single prompt feature. Ignore outlier prompt features. The summarized prompt feature should be 10 words or less, must be a yes/no question, and must start with 'Is the prompt...'.
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
            # DEFAULT_GENERATOR_PROMPT_V1,
            DEFAULT_GENERATOR_PROMPT_V2,
        ]
    )
    prompt_generator_prompts: list[str] = field(
        default_factory=lambda: [
            # DEFAULT_PROMPT_GENERATOR_PROMPT_V1,
            # DEFAULT_PROMPT_GENERATOR_PROMPT_V2,
        ]
    )
    voting_prompt: str = DEFAULT_VOTING_PROMPT
    prompt_voting_prompt: str = DEFAULT_PROMPT_VOTING_PROMPT
    cluster_summary_prompt: str = DEFAULT_CLUSTER_SUMMARY_PROMPT
    prompt_cluster_summary_prompt: str = DEFAULT_PROMPT_CLUSTER_SUMMARY_PROMPT
