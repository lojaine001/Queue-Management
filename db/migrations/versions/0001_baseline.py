"""Baseline: the database structure exactly as production had it on 2026-11-13.

Copied from `pg_dump --schema-only` of the production database (PostgreSQL 17.9,
TimescaleDB 2.26.2), not from the programs' start-up code, which had drifted from it
(for example the checkout camera writes service_events.lane_id but never created it).

- A new, empty database: `alembic -c db/alembic.ini upgrade head` creates everything.
- A database that already has these tables (production): mark it instead with
  `alembic -c db/alembic.ini stamp 0001_baseline`; running this migration there fails
  on the first existing table and changes nothing.

Revision ID: 0001_baseline
Revises:
Create Date: 2026-11-13
"""
from alembic import op

revision = "0001_baseline"
down_revision = None
branch_labels = None
depends_on = None


TABLES = """
CREATE EXTENSION IF NOT EXISTS timescaledb WITH SCHEMA public;

-- ── Camera data (TimescaleDB hypertables, 7-day chunks) ─────────────────────
CREATE TABLE public.entrance_events (
    id integer NOT NULL,
    track_id integer,
    gender character varying(20),
    age_estimate numeric(5,1),
    confidence numeric(5,4),
    camera_id character varying(50),
    "timestamp" timestamp with time zone DEFAULT now() NOT NULL,
    dwell_seconds numeric(8,2),
    has_bag boolean DEFAULT false,
    has_caddy boolean DEFAULT false,
    is_group boolean DEFAULT false,
    group_id character varying(100),
    active_head_tracks_in_lane integer DEFAULT 0,
    equipment_type character varying(30) DEFAULT 'none'::character varying
);
CREATE SEQUENCE public.entrance_events_id_seq AS integer START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;
ALTER SEQUENCE public.entrance_events_id_seq OWNED BY public.entrance_events.id;
ALTER TABLE ONLY public.entrance_events ALTER COLUMN id SET DEFAULT nextval('public.entrance_events_id_seq'::regclass);

CREATE TABLE public.queue_state_snapshots (
    id bigint NOT NULL,
    "timestamp" timestamp with time zone DEFAULT now() NOT NULL,
    camera_id character varying(100),
    queue_count integer DEFAULT 0 NOT NULL,
    avg_dwell_sec double precision DEFAULT 0,
    max_dwell_sec double precision DEFAULT 0,
    active_lanes integer DEFAULT 2,
    lane_counts jsonb
);
CREATE SEQUENCE public.queue_state_snapshots_id_seq START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;
ALTER SEQUENCE public.queue_state_snapshots_id_seq OWNED BY public.queue_state_snapshots.id;
ALTER TABLE ONLY public.queue_state_snapshots ALTER COLUMN id SET DEFAULT nextval('public.queue_state_snapshots_id_seq'::regclass);

CREATE TABLE public.service_events (
    id bigint NOT NULL,
    "timestamp" timestamp with time zone DEFAULT now() NOT NULL,
    camera_id character varying(100),
    track_id integer,
    lane_id integer,
    total_dwell_sec double precision DEFAULT 0,
    equipment_type character varying(30) DEFAULT 'none'::character varying,
    active_head_tracks_in_lane integer
);
CREATE SEQUENCE public.service_events_id_seq START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;
ALTER SEQUENCE public.service_events_id_seq OWNED BY public.service_events.id;
ALTER TABLE ONLY public.service_events ALTER COLUMN id SET DEFAULT nextval('public.service_events_id_seq'::regclass);

CREATE TABLE public.lane_events (
    id bigint NOT NULL,
    "timestamp" timestamp with time zone DEFAULT now() NOT NULL,
    camera_id character varying(100),
    active_lanes integer NOT NULL,
    note text
);
CREATE SEQUENCE public.lane_events_id_seq START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;
ALTER SEQUENCE public.lane_events_id_seq OWNED BY public.lane_events.id;
ALTER TABLE ONLY public.lane_events ALTER COLUMN id SET DEFAULT nextval('public.lane_events_id_seq'::regclass);

-- ── Forecasts ───────────────────────────────────────────────────────────────
CREATE TABLE public.queue_predictions (
    id bigint NOT NULL,
    predicted_at timestamp with time zone NOT NULL,
    prediction_for timestamp with time zone NOT NULL,
    prophet_yhat numeric(8,2),
    lstm_yhat numeric(8,2),
    xgb_yhat numeric(8,2),
    ensemble_yhat numeric(8,2),
    est_wait_minutes numeric(8,2),
    wait_15m numeric(8,2),
    wait_30m numeric(8,2),
    status character varying(10),
    wait_45m numeric(8,2),
    wait_1lane_15m numeric(8,2),
    wait_2lane_15m numeric(8,2),
    wait_3lane_15m numeric(8,2),
    pipeline_source character varying(20),
    source character varying(20) DEFAULT 'ensemble'::character varying
);
CREATE SEQUENCE public.queue_predictions_id_seq START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;
ALTER SEQUENCE public.queue_predictions_id_seq OWNED BY public.queue_predictions.id;
ALTER TABLE ONLY public.queue_predictions ALTER COLUMN id SET DEFAULT nextval('public.queue_predictions_id_seq'::regclass);
ALTER TABLE ONLY public.queue_predictions ADD CONSTRAINT queue_predictions_prediction_for_key UNIQUE (prediction_for);

CREATE TABLE public.forecast_runs (
    id bigint NOT NULL,
    run_id text NOT NULL,
    run_started_at timestamp with time zone NOT NULL,
    data_cutoff timestamp with time zone,
    forecast_origin timestamp with time zone NOT NULL,
    published_at timestamp with time zone NOT NULL,
    model_version text NOT NULL,
    source text NOT NULL,
    predictions jsonb NOT NULL
);
ALTER TABLE public.forecast_runs ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.forecast_runs_id_seq START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1
);
ALTER TABLE ONLY public.forecast_runs ADD CONSTRAINT forecast_runs_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.forecast_runs ADD CONSTRAINT forecast_runs_run_id_key UNIQUE (run_id);
CREATE INDEX forecast_runs_published_idx ON public.forecast_runs USING btree (published_at DESC);

CREATE TABLE public.wait_forecast_runs (
    publication_id text NOT NULL,
    published_at timestamp with time zone DEFAULT now() NOT NULL,
    settings_revision bigint NOT NULL,
    selected_lanes integer NOT NULL,
    payload jsonb NOT NULL
);
ALTER TABLE ONLY public.wait_forecast_runs ADD CONSTRAINT wait_forecast_runs_pkey PRIMARY KEY (publication_id);
CREATE INDEX wait_forecast_runs_published_idx ON public.wait_forecast_runs USING btree (published_at DESC);

CREATE TABLE public.forecast_settings (
    id integer DEFAULT 1 NOT NULL,
    revision bigint DEFAULT 1 NOT NULL,
    config jsonb NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT forecast_settings_id_check CHECK ((id = 1))
);
ALTER TABLE ONLY public.forecast_settings ADD CONSTRAINT forecast_settings_pkey PRIMARY KEY (id);

-- ── Dashboard ───────────────────────────────────────────────────────────────
CREATE TABLE public.dashboard_state (
    id integer DEFAULT 1 NOT NULL,
    updated_at timestamp with time zone DEFAULT now(),
    queue_now integer,
    service_min double precision,
    wait_0m double precision,
    wait_5m double precision,
    wait_10m double precision,
    wait_15m double precision,
    lane1_wait_15m double precision,
    lane2_wait_15m double precision,
    lane3_wait_15m double precision,
    lane4_wait_15m double precision,
    open_lanes integer,
    demographics_json text,
    entries_hour_json text,
    forecast_json jsonb
);
ALTER TABLE ONLY public.dashboard_state ADD CONSTRAINT dashboard_state_pkey PRIMARY KEY (id);

CREATE TABLE public.dashboard_statistics (
    id integer NOT NULL,
    updated_at timestamp with time zone NOT NULL,
    camera_id text NOT NULL,
    demographics jsonb NOT NULL,
    entries_hour jsonb NOT NULL,
    CONSTRAINT dashboard_statistics_id_check CHECK ((id = 1))
);
ALTER TABLE ONLY public.dashboard_statistics ADD CONSTRAINT dashboard_statistics_pkey PRIMARY KEY (id);

-- ── Web app ─────────────────────────────────────────────────────────────────
CREATE TABLE public.push_subscriptions (
    endpoint text NOT NULL,
    p256dh text NOT NULL,
    auth text NOT NULL,
    threshold_min double precision NOT NULL,
    horizon_min integer NOT NULL,
    was_over boolean DEFAULT false NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL
);
ALTER TABLE ONLY public.push_subscriptions ADD CONSTRAINT push_subscriptions_pkey PRIMARY KEY (endpoint);

CREATE TABLE public.alert_responses (
    id integer NOT NULL,
    response text NOT NULL,
    lane_id text,
    responded_at timestamp with time zone DEFAULT now()
);
CREATE SEQUENCE public.alert_responses_id_seq AS integer START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;
ALTER SEQUENCE public.alert_responses_id_seq OWNED BY public.alert_responses.id;
ALTER TABLE ONLY public.alert_responses ALTER COLUMN id SET DEFAULT nextval('public.alert_responses_id_seq'::regclass);
ALTER TABLE ONLY public.alert_responses ADD CONSTRAINT alert_responses_pkey PRIMARY KEY (id);

-- ── Legacy: no code uses it (kept so the baseline matches production) ───────
CREATE TABLE public.queue_events (
    id integer NOT NULL,
    "timestamp" timestamp without time zone DEFAULT now(),
    track_id integer,
    camera_id character varying(100),
    confidence double precision,
    dwell_seconds double precision DEFAULT 0
);
CREATE SEQUENCE public.queue_events_id_seq AS integer START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;
ALTER SEQUENCE public.queue_events_id_seq OWNED BY public.queue_events.id;
ALTER TABLE ONLY public.queue_events ALTER COLUMN id SET DEFAULT nextval('public.queue_events_id_seq'::regclass);
ALTER TABLE ONLY public.queue_events ADD CONSTRAINT queue_events_pkey PRIMARY KEY (id);
"""

# create_hypertable also creates the "<table>_<column>_idx" (column DESC) index that
# production has on each of these tables, except where the time column already has a
# unique constraint (queue_predictions): that one is created explicitly afterwards.
HYPERTABLES = [
    ("entrance_events", "timestamp"),
    ("queue_state_snapshots", "timestamp"),
    ("service_events", "timestamp"),
    ("lane_events", "timestamp"),
    ("queue_predictions", "prediction_for"),
]


def upgrade() -> None:
    op.execute(TABLES)
    for table, column in HYPERTABLES:
        op.execute(
            f"SELECT create_hypertable('public.{table}', '{column}', "
            f"chunk_time_interval => INTERVAL '7 days')"
        )
    op.execute(
        "CREATE INDEX IF NOT EXISTS queue_predictions_prediction_for_idx "
        "ON public.queue_predictions USING btree (prediction_for DESC)"
    )


def downgrade() -> None:
    raise RuntimeError("The baseline is not reversible: it would drop every table and its data.")
