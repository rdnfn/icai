_PRINCIPLES_DATA_V1 = []  # TODO: add principle v1

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
