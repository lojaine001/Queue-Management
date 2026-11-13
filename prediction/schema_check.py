"""The database structure comes from the migrations in db/, not from the programs.

Programs only check that the tables they use exist, so a database that was never
migrated fails at start-up with a message saying what to run, instead of with a
"relation does not exist" error at the first query.
"""

MIGRATE_HINT = ("run `alembic -c db/alembic.ini upgrade head` from the project root "
                "(on Docker the migrate service does it); see db/README.md")


def missing_tables(cursor, tables) -> list[str]:
    cursor.execute(
        "SELECT t FROM unnest(%s::text[]) AS t WHERE to_regclass('public.' || t) IS NULL",
        (list(tables),),
    )
    return [row["t"] if isinstance(row, dict) else row[0] for row in cursor.fetchall()]


def require_tables(cursor, *tables: str) -> None:
    missing = missing_tables(cursor, tables)
    if missing:
        raise RuntimeError(f"Database table(s) missing: {', '.join(missing)}; {MIGRATE_HINT}")
