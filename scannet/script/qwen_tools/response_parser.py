import json
import re

def validate_categories(items, field_name):
    """Validate category records without inventing or merging descriptions."""
    if not isinstance(items, list):
        raise ValueError(f"{field_name} must be a list")

    result = []

    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"{field_name}[{index}] must be an object")

        name = item.get("name")
        description = item.get("description")

        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{field_name}[{index}] has an invalid name")

        if not isinstance(description, str) or not description.strip():
            raise ValueError(
                f"{field_name}[{index}] has an invalid description"
            )

        name = " ".join(name.split()).lower()
        description = description.strip()

        # Downstream code uses a colon to separate names and descriptions.
        if ":" in name:
            raise ValueError(f"Category name must not contain a colon: {name}")

        result.append({
            "name": name,
            "description": description,
        })

    return result

def decode_model_json(content):
    """Decode JSON with optional BOM or a single complete Markdown fence."""
    text = content.strip().lstrip("\ufeff").strip()

    if text.startswith("```"):
        match = re.fullmatch(
            r"```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```",
            text,
            flags=re.IGNORECASE,
        )

        if match is None:
            raise ValueError(
                "Expected a single complete JSON code block"
            )

        text = match.group(1).strip()

    return json.loads(text)


def parse_response(record):
    """Extract validated categories from a saved API response."""
    response = record.get("response")
    if not isinstance(response, dict):
        raise ValueError("Missing response object")

    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("Expected exactly one response choice")

    choice = choices[0]

    # Reject truncated or otherwise incomplete responses.
    if choice.get("finish_reason") != "stop":
        raise ValueError(
            f"Response did not finish normally: "
            f"{choice.get('finish_reason')!r}"
        )

    message = choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("Missing response message")

    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Response content is empty")

    # Invalid JSON remains available in the saved raw response.
    try:
        payload = decode_model_json(content)
    except ValueError as exc:
        frame_id = record.get("frame_id", "<unknown>")
        raise ValueError(
            f"Frame {frame_id}: model content is not valid JSON. "
            f"Beginning: {content[:160]!r}. "
            "The raw response has been preserved."
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError("Model output must be a JSON object")

    objects = validate_categories(payload.get("objects"), "objects")

    surfaces = validate_categories(
        payload.get("surface_categories", []),
        "surface_categories",
    )

    return {
        "objects": objects,
        "surface_categories": surfaces,
    }