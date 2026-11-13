"""Alembic environment: connects with the same DB_* settings as the rest of IQMS."""
import os
from pathlib import Path

from alembic import context
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

try:
    from dotenv import load_dotenv
    # The project's single .env; values already set in the environment (Docker) win.
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
except ImportError:
    pass


def _url() -> URL:
    password = os.getenv("DB_PASSWORD")
    if not password:
        raise SystemExit("DB_PASSWORD is not set (expected in .env or the environment).")
    return URL.create(
        "postgresql+psycopg2",
        username=os.getenv("DB_USER", "postgres"),
        password=password,
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 5432)),
        database=os.getenv("DB_NAME", "iqms"),
    )


def run_migrations_online() -> None:
    engine = create_engine(_url())
    with engine.connect() as connection:
        context.configure(connection=connection, transaction_per_migration=True)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    raise SystemExit("Offline mode is not used; run against the database.")
run_migrations_online()
