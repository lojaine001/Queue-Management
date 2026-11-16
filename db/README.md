# Database migrations

The database structure (tables, columns, indexes, TimescaleDB hypertables) is defined
here, in numbered migrations under `migrations/versions/`, and applied with
[Alembic](https://alembic.sqlalchemy.org). The database records which migrations it has
(table `alembic_version`), so each one runs once, in order, on every machine.

Commands run from the project root and use the `DB_*` settings from `.env`:

| Task | Command |
|---|---|
| Apply every migration not yet applied | `alembic -c db/alembic.ini upgrade head` |
| Show the version the database is at | `alembic -c db/alembic.ini current` |
| List the migrations | `alembic -c db/alembic.ini history` |

On Docker the `migrate` service runs `upgrade head` on every start, before the other
services (see `deploy/docker/README.md`).

## The baseline and production

`0001_baseline` is production's structure exactly as it was on 2026-11-13. A new, empty
database gets it with `upgrade head`. A database that already had these tables, like the
production one, is **marked** instead of migrated, once:
```
alembic -c db/alembic.ini stamp 0001_baseline
```
This only creates the `alembic_version` table; no other table or row changes. Running
`upgrade head` on an unmarked database that already has the tables fails on the first
table ("already exists") and changes nothing.

## Migrations so far

| Revision | What it does |
|---|---|
| `0001_baseline` | Production's structure on 2026-11-13 (13 tables, 5 hypertables) |
| `0002_config_changes` | Table `config_changes`: who changed which setting (open lanes, forecast settings), when, old and new value. Written by `prediction/config_audit.py` in the same transaction as the change; shown in the dashboard ("Configuration change history") |

On the Windows machine, apply a new migration **before** restarting with the code that
needs it: `alembic -c db/alembic.ini upgrade head`, then `.\start_all.ps1`.
`/health/ready` says which version the database is at and which one the code expects.

## Changing the structure

1. `alembic -c db/alembic.ini revision -m "add lane_id to alert_responses"` creates the
   next numbered file in `migrations/versions/`.
2. Write the change in plain SQL with `op.execute(...)` in `upgrade()`, and its reverse in
   `downgrade()` when it can be undone without losing data.
3. Test it on a copy, not on production: an empty database (`upgrade head` from nothing)
   and a copy of production's data.
4. Commit it with the code that needs it. Each machine applies it with `upgrade head`
   (automatic on Docker).

The programs do not create or change tables themselves; a new column goes in a migration.
