"""The 'rows of one day' condition shared by the day recap's queries.

The day is passed as a query parameter, never written into the SQL text. `column` is
always a column name chosen in the code, never user input."""
from __future__ import annotations


def on_day_or_since_today(column: str, day) -> tuple[str, tuple]:
    """(condition, parameters): rows of the store-local `day`; with no day, rows since
    today 00:00 (the recap's fallback when the database has no entrance data at all)."""
    if day is None:
        return f"{column} >= CURRENT_DATE", ()
    return f"DATE({column}) = %s", (day,)
