def is_col_field(field: str | None) -> bool:
    """Checks if a string field is formatted as "col(column_name)".

    This format is used to denote a column in a Polars DataFrame in the event conversion configuration.

    Args:
        field (str | None): The field to check.

    Returns:
        bool: True if the field is formatted as "col(column_name)", False otherwise.

    Examples:
        >>> is_col_field("col(subject_id)")
        True
        >>> is_col_field("col(subject_id")
        False
        >>> is_col_field("subject_id)")
        False
        >>> is_col_field("column(subject_id)")
        False
        >>> is_col_field("subject_id")
        False
        >>> is_col_field(None)
        False
    """
    if field is None:
        return False
    return field.startswith("col(") and field.endswith(")")


def parse_col_field(field: str) -> str:
    """Extracts the actual column name from a string formatted as "col(column_name)".

    Args:
        field (str): A string formatted as "col(column_name)".

    Raises:
        ValueError: If the input string does not match the expected format.

    Examples:
        >>> parse_col_field("col(subject_id)")
        'subject_id'
        >>> parse_col_field("col(subject_id")
        Traceback (most recent call last):
        ...
        ValueError: Invalid column field: col(subject_id
        >>> parse_col_field("column(subject_id)")
        Traceback (most recent call last):
        ...
        ValueError: Invalid column field: column(subject_id)
    """
    if not is_col_field(field):
        raise ValueError(f"Invalid column field: {field}")
    return field[4:-1]
