"""Utility functions and constants for plotting."""

HOVER_TEMPLATE = (
    "<b>Result:</b> {vote} <br><b>Principle:</b> {principle_str}<br>{text_columns_str}"
)


def get_string_with_breaks(
    text: str,
    line_length: int = 50,
    max_lines: int = 20,
    add_spaces_at_end: bool = False,
) -> str:
    text = text.strip()
    text = text.replace("\n", " <br> ")
    lines = text.split(" ")

    full_text = ""
    current_line = ""
    lines_so_far = 0

    for word in lines:
        if len(current_line) + len(word) > line_length or word == "<br>":

            if not add_spaces_at_end:
                spacing = ""
            else:
                spacing = " " * (line_length - len(current_line))

            full_text += current_line + spacing + "<br>"
            current_line = ""
            lines_so_far += 1
            if lines_so_far >= max_lines:
                current_line += "..." + spacing
                break

        if word != "<br>":
            current_line += word + " "

    full_text += current_line

    return full_text


def merge_into_columns(text_1, text_2):

    merged_text = ""

    text_1_lines = text_1.split("<br>")
    text_2_lines = text_2.split("<br>")
    max_lines = max(len(text_1_lines), len(text_2_lines))
    col_width = max([len(line) for line in text_1_lines + text_2_lines])

    for i in range(max_lines):
        line_1 = text_1_lines[i] if i < len(text_1_lines) else ""
        line_2 = text_2_lines[i] if i < len(text_2_lines) else ""

        line_1_without_html = (
            line_1.replace("<br>", "").replace("<b>", "").replace("</b>", "")
        )

        if len(line_1_without_html) < col_width:
            line_1 += " " * (col_width - len(line_1_without_html))

        merged_text += f"{line_1} | {line_2}<br>"

    return merged_text
