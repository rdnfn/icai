_PRINCIPLES_DATA_V1 = [
    {
        "info": "Principles generated with Inverse Constitutional AI",
        "principles_short_version": [
            -"Select the response that provides more detailed explanations."
            - "Select the response that is overly lengthy and lacks brevity."
            - "Select the response that provides a numbered list format."
            - "Select the response that contains redundant information."
            - "Select the response that promotes constructive dialogue over blame."
            - "Select the response that lacks neutrality in political matters."
            - "Select the response that presents a definitive stance without nuance."
            - "Select the response that provides a scientific explanation."
            - "Select the response that avoids acknowledging complexity of the issue."
            - "Select the response that introduces ambiguity about the assistant's nature."
            - "Select the response that emphasizes neutrality over providing information."
            - "Select the response that lacks nuance in political analysis."
            - "Select the response that avoids any suggestion of illegal activity."
            - "Select the response that promotes divisive political statements."
            - "Select the response that introduces unnecessary philosophical debate."
            - "Select the response that acknowledges limitations in available information."
            - "Select the response that includes negative character judgments."
            - "Select the response that promotes conspiracy theories without critical analysis."
            - "Select the response that includes inappropriate language or content."
            - "Select the response that lacks a clear stance on immigration."
            - "Select the response that presents ambiguous or non-committal language."
            - "Select the response that highlights challenges faced by refugees."
            - "Select the response that depicts distressing experiences in detail."
            - "Select the response that assigns sole blame without context."
            - "Select the response that promotes unrealistic expectations for quick money."
            - "Select the response that lacks credible historical evidence."
            - "Select the response that avoids providing a speculative answer."
            - "Select the response that includes unrelated product listings."
            - "Select the response that perpetuates harmful stereotypes about race."
            - "Select the response that does not consider personal preferences."
            - "Select the response that respects privacy and consent."
            - "Select the response that uses more formal language."
            - "Select the response that presents a one-sided argument."
            - "Select the response that presents a biased viewpoint without nuance."
            - "Select the response that lacks a clear stance on morality."
            - "Select the response that lacks respect for diverse beliefs."
            - "Select the response that provides an apology instead of an answer."
            - "Select the response that feels more casual and friendly."
            - "Select the response that is more concise and structured."
            - "Select the response that is overly general and vague."
        ],
    },
]

_PRINCIPLES_DATA_V2 = [
    {
        "info": "Principles generated with Inverse Constitutional AI (partly adapted)",
        "principles_short_version": [
            "is more concise",
            "is too long",
            "is more verbose",
            "provides a numbered list format",
            "has more structured formatting",
            "ends with a follow-up question",
            "more directly follows instructions",
            "more strictly follows the requested format",
            "is more polite",
            "has a more friendly tone",
            "uses more casual language",
            "uses more formal language",
            "provides more detailed explanations",
            "includes inappropriate language",
            "suggests illegal activities",
            "has a more avoidant tone",
            "is less complex",
            "is more factually correct",
            "follows best practices",
            "is more offensive",
        ],
    },
    {
        "info": "Principles inspired by VibeCheck paper (v6) (https://arxiv.org/abs/2410.12851)",
        "principles_short_version": [
            "includes more references to other sources",
            "expresses more emotion",
            "contains less harmful information",
            "refuses to answer the question",
            "uses more bold and italics text",
            "provides more examples",
            "uses more humour",
            "uses more personal pronouns (I, we, you)",
            "includes more ethical considerations",
            "acknowledges own limitations or uncertainty more",
            "is more creative and original",
            "makes more confident statements",
            "provides clearer reasoning with well-supported arguments",
            "provides conclusions without full reasoning",
            "actively engages the reader with rhetorical questions",
            "is more vague",
            "uses a more enthusiastic tone",
            "contains more concise straightforward solution steps",
            "avoids unnecessary repetition",
            "uses more mathematical symbols and notation",
        ],
    },
]


def _get_principles_from_data(data: list[dict]) -> list[str]:
    return [
        f"Select the response that {principle}"
        for subset in data
        for principle in subset["principles_short_version"]
    ]


DEFAULT_PRINCIPLES = {
    "v1": _get_principles_from_data(_PRINCIPLES_DATA_V1),
    "v2": _get_principles_from_data(_PRINCIPLES_DATA_V2),
}
