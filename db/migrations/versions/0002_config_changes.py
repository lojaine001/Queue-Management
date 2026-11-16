"""Configuration change log: who changed which setting, when, from what to what.

One row per changed setting: the open lanes (API /set-lanes or the dashboard's lane
selector) and every forecast setting saved from the dashboard. Written in the same
transaction as the change itself (prediction/config_audit.py), so a change and its row
are saved together or not at all.

"Who" is what the system can tell without user accounts: the source (api / dashboard)
and an actor description (API key + address + app, or dashboard + browser address +
Windows user). Small table (a few rows per day): a plain table, not a hypertable.

Revision ID: 0002_config_changes
Revises: 0001_baseline
Create Date: 2026-11-16
"""
from alembic import op

revision = "0002_config_changes"
down_revision = "0001_baseline"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE public.config_changes (
            id bigserial PRIMARY KEY,
            changed_at timestamp with time zone DEFAULT now() NOT NULL,
            source text NOT NULL CHECK (source IN ('api', 'dashboard')),
            actor text,
            setting text NOT NULL,
            old_value jsonb,
            new_value jsonb,
            settings_revision bigint
        )
    """)
    op.execute("CREATE INDEX config_changes_changed_at_idx ON public.config_changes (changed_at DESC)")


def downgrade() -> None:
    # Drops the history itself; only for undoing a fresh deploy.
    op.execute("DROP TABLE public.config_changes")
