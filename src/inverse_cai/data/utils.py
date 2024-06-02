"""Data utils for manipulating standard preference data."""


def get_preferred_text(row):
    if row["preferred_text"] == "text_a":
        return row["text_a"]
    elif row["preferred_text"] == "text_b":
        return row["text_b"]
    else:
        raise ValueError(f"Invalid preferred_text value: {row['preferred_text']}")


def get_rejected_text(row):
    if row["preferred_text"] == "text_a":
        return row["text_b"]
    elif row["preferred_text"] == "text_b":
        return row["text_a"]
    else:
        raise ValueError(f"Invalid preferred_text value: {row['preferred_text']}")
