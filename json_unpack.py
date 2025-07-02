import json

def unpack_json_to_string(data, indent=0):
    """
    Unpacks JSON data (dictionaries and lists) into a human-readable string format,
    handling nested structures with indentation.

    Args:
        data: The JSON data (dictionary or list) to unpack.
        indent: The current indentation level (for internal recursive calls).

    Returns:
        A string representation of the unpacked JSON data.
    """
    indent_str = "    " * indent  # 4 spaces per indent level
    output_lines = []

    if isinstance(data, dict):
        # If it's a dictionary, iterate through its key-value pairs
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                # If the value is another dictionary or a list, recurse
                output_lines.append(f"{indent_str}{key}:")
                output_lines.append(unpack_json_to_string(value, indent + 1))
            else:
                # Otherwise, it's a simple key-value pair
                output_lines.append(f"{indent_str}{key}: {value}")
    elif isinstance(data, list):
        # If it's a list, iterate through its items
        for i, item in enumerate(data):
            if isinstance(item, (dict, list)):
                # If the item is a dictionary or a list, recurse
                output_lines.append(f"{indent_str}- Item {i + 1}:")
                output_lines.append(unpack_json_to_string(item, indent + 1))
            else:
                # Otherwise, it's a simple list item
                output_lines.append(f"{indent_str}- {item}")
    else:
        # For non-dict/list data types (e.g., strings, numbers, booleans)
        output_lines.append(f"{indent_str}{data}")

    return "\n".join(output_lines)