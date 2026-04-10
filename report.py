"""
IQMS - Intelligent Queue Management System
Report Generator
Queries both entrance_events and queue_events from PostgreSQL
and produces a self-contained HTML report with charts.

Usage:
    python report.py
Output:
    iqms_report.html  (open in any browser)
"""

import os
import base64
import warnings
import psycopg2
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from io import BytesIO
from datetime import datetime

warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────────────
DB_CONFIG = dict(host="localhost", port=5432, dbname="iqms",
                 user="postgres", password="0000")
OUTPUT_FILE = "iqms_report.html"
STORE_OPEN  = 8   # hour
STORE_CLOSE = 21  # hour


# ── HELPERS ───────────────────────────────────────────────────────────────────
def connect():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("[DB] Connected ✓")
        return conn
    except Exception as e:
        print(f"[DB] ERROR: {e}")
        raise


def fig_to_b64(fig):
    """Convert matplotlib figure to base64 PNG string for embedding in HTML."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return encoded


STYLE = {
    'bg':      '#0f1117',
    'card':    '#1a1d27',
    'accent':  '#e8663d',
    'accent2': '#4fc3f7',
    'text':    '#e0e0e0',
    'muted':   '#888',
    'grid':    '#2a2d3a',
}

def set_dark_style():
    plt.rcParams.update({
        'figure.facecolor':  STYLE['bg'],
        'axes.facecolor':    STYLE['card'],
        'axes.edgecolor':    STYLE['grid'],
        'axes.labelcolor':   STYLE['text'],
        'axes.titlecolor':   STYLE['text'],
        'xtick.color':       STYLE['muted'],
        'ytick.color':       STYLE['muted'],
        'grid.color':        STYLE['grid'],
        'text.color':        STYLE['text'],
        'legend.facecolor':  STYLE['card'],
        'legend.edgecolor':  STYLE['grid'],
        'font.family':       'DejaVu Sans',
    })

set_dark_style()


# ── DATA LOADING ──────────────────────────────────────────────────────────────
def load_entrance(conn):
    q = """
        SELECT id, timestamp, track_id, gender, age_estimate,
               confidence, camera_id, dwell_seconds
        FROM entrance_events
        ORDER BY timestamp
    """
    df = pd.read_sql(q, conn, parse_dates=['timestamp'])
    if df.empty:
        return df
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
    df['hour']      = df['timestamp'].dt.hour
    df['date']      = df['timestamp'].dt.date
    df['day_name']  = df['timestamp'].dt.day_name()
    df['gender']    = df['gender'].str.strip().str.lower()
    return df


def load_queue(conn):
    q = """
        SELECT id, timestamp, track_id, camera_id,
               confidence, dwell_seconds
        FROM queue_events
        ORDER BY timestamp
    """
    df = pd.read_sql(q, conn, parse_dates=['timestamp'])
    if df.empty:
        return df
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
    df['hour']      = df['timestamp'].dt.hour
    df['date']      = df['timestamp'].dt.date
    return df


# ── CHARTS ────────────────────────────────────────────────────────────────────
def chart_daily_visitors(df):
    daily = df.groupby('date').size().reset_index(name='count')
    fig, ax = plt.subplots(figsize=(10, 3.5))
    bars = ax.bar(range(len(daily)), daily['count'],
                  color=STYLE['accent'], alpha=0.85, width=0.6)
    ax.set_xticks(range(len(daily)))
    ax.set_xticklabels([str(d) for d in daily['date']],
                       rotation=30, ha='right', fontsize=8)
    ax.set_title('Daily Visitor Count', fontsize=13, pad=10)
    ax.set_ylabel('Visitors')
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, daily['count']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(val), ha='center', va='bottom', fontsize=8,
                color=STYLE['text'])
    fig.tight_layout()
    return fig_to_b64(fig)


def chart_hourly_heatmap(df):
    days_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    pivot = df.groupby(['day_name','hour']).size().unstack(fill_value=0)
    pivot = pivot.reindex([d for d in days_order if d in pivot.index])
    hours = list(range(STORE_OPEN, STORE_CLOSE + 1))
    pivot = pivot.reindex(columns=hours, fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 3.5))
    im = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd',
                   interpolation='nearest')
    ax.set_xticks(range(len(hours)))
    ax.set_xticklabels([f"{h:02d}:00" for h in hours], rotation=45,
                       ha='right', fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_title('Traffic Heatmap — Day × Hour', fontsize=13, pad=10)
    plt.colorbar(im, ax=ax, label='Visitors')
    fig.tight_layout()
    return fig_to_b64(fig)


def chart_hourly_avg(df):
    hourly = df.groupby('hour').size()
    days   = df['date'].nunique() or 1
    hourly = hourly / days
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.fill_between(hourly.index, hourly.values,
                    alpha=0.25, color=STYLE['accent'])
    ax.plot(hourly.index, hourly.values, color=STYLE['accent'],
            linewidth=2.5, marker='o', markersize=5)
    ax.set_xticks(range(STORE_OPEN, STORE_CLOSE + 1))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(STORE_OPEN, STORE_CLOSE + 1)],
                       rotation=45, ha='right', fontsize=8)
    ax.set_title('Average Visitors per Hour (across all days)', fontsize=13, pad=10)
    ax.set_ylabel('Avg visitors / hour')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig_to_b64(fig)


def chart_gender(df):
    gender_map = {'male': 'Male', 'female': 'Female',
                  'm': 'Male', 'f': 'Female', 'unknown': 'Unknown'}
    df = df.copy()
    df['gender'] = df['gender'].map(lambda x: gender_map.get(x, 'Unknown'))
    counts = df['gender'].value_counts()
    colors = [STYLE['accent'], STYLE['accent2'], STYLE['muted']]
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=counts.index,
        autopct='%1.1f%%', colors=colors[:len(counts)],
        startangle=140, pctdistance=0.75,
        wedgeprops=dict(width=0.55))
    for t in autotexts:
        t.set_color(STYLE['bg'])
        t.set_fontsize(10)
    ax.set_title('Gender Distribution', fontsize=13, pad=10)
    fig.tight_layout()
    return fig_to_b64(fig)


def chart_age(df):
    ages = df['age_estimate'].dropna()
    ages = ages[(ages > 0) & (ages < 100)]
    if ages.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(ages, bins=20, color=STYLE['accent2'], alpha=0.8, edgecolor=STYLE['bg'])
    ax.axvline(ages.mean(), color=STYLE['accent'], linewidth=2,
               linestyle='--', label=f'Mean: {ages.mean():.1f}')
    ax.set_title('Age Distribution of Visitors', fontsize=13, pad=10)
    ax.set_xlabel('Estimated Age')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig_to_b64(fig)


def chart_dwell_entrance(df):
    dwell = df['dwell_seconds'].dropna()
    dwell = dwell[(dwell > 0) & (dwell < 300)]
    if dwell.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(dwell, bins=30, color=STYLE['accent'], alpha=0.8,
            edgecolor=STYLE['bg'])
    ax.axvline(dwell.mean(), color=STYLE['accent2'], linewidth=2,
               linestyle='--', label=f'Mean: {dwell.mean():.1f}s')
    ax.set_title('Entrance Zone Dwell Time Distribution', fontsize=13, pad=10)
    ax.set_xlabel('Seconds in zone')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig_to_b64(fig)


def chart_checkout_service(df):
    dwell = df['dwell_seconds'].dropna()
    dwell = dwell[(dwell > 0) & (dwell < 600)]
    if dwell.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(dwell, bins=30, color=STYLE['accent2'], alpha=0.8,
            edgecolor=STYLE['bg'])
    ax.axvline(dwell.mean(), color=STYLE['accent'], linewidth=2,
               linestyle='--', label=f'Mean: {dwell.mean():.1f}s')
    ax.set_title('Checkout Service Time Distribution', fontsize=13, pad=10)
    ax.set_xlabel('Seconds at checkout')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig_to_b64(fig)


def chart_checkout_hourly(df):
    hourly = df.groupby('hour')['dwell_seconds'].mean().dropna()
    if hourly.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.bar(hourly.index, hourly.values / 60,
           color=STYLE['accent2'], alpha=0.8, width=0.6)
    ax.set_title('Average Checkout Service Time by Hour', fontsize=13, pad=10)
    ax.set_xlabel('Hour of day')
    ax.set_ylabel('Avg service time (min)')
    ax.set_xticks(hourly.index)
    ax.set_xticklabels([f"{h:02d}:00" for h in hourly.index],
                       rotation=45, ha='right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    return fig_to_b64(fig)


# ── STATS ─────────────────────────────────────────────────────────────────────
def compute_stats(ent, que):
    stats = {}

    if not ent.empty:
        stats['total_visitors']    = len(ent)
        stats['days_monitored']    = ent['date'].nunique()
        stats['avg_daily']         = round(len(ent) / max(stats['days_monitored'], 1), 1)
        stats['date_from']         = str(ent['timestamp'].min().date())
        stats['date_to']           = str(ent['timestamp'].max().date())

        peak_hour_series = ent.groupby('hour').size()
        stats['peak_hour']         = int(peak_hour_series.idxmax())
        stats['peak_hour_count']   = int(peak_hour_series.max())

        gender_counts = ent['gender'].value_counts()
        stats['gender_breakdown']  = gender_counts.to_dict()

        ages = ent['age_estimate'].dropna()
        ages = ages[(ages > 0) & (ages < 100)]
        stats['avg_age']           = round(ages.mean(), 1) if not ages.empty else 'N/A'

        dwell = ent['dwell_seconds'].dropna()
        dwell = dwell[dwell > 0]
        stats['avg_dwell_entrance'] = round(dwell.mean(), 1) if not dwell.empty else 'N/A'
    else:
        stats['total_visitors'] = 0

    if not que.empty:
        stats['total_checkout_events'] = len(que)
        dwell = que['dwell_seconds'].dropna()
        dwell = dwell[dwell > 0]
        stats['avg_service_time']  = round(dwell.mean(), 1) if not dwell.empty else 'N/A'
        stats['max_service_time']  = round(dwell.max(), 1)  if not dwell.empty else 'N/A'

        peak_q = que.groupby('hour').size()
        stats['checkout_peak_hour'] = int(peak_q.idxmax()) if not peak_q.empty else 'N/A'
    else:
        stats['total_checkout_events'] = 0

    return stats


# ── HTML BUILDER ──────────────────────────────────────────────────────────────
def img_tag(b64):
    if b64 is None:
        return '<p style="color:#888;font-style:italic;">Not enough data yet.</p>'
    return f'<img src="data:image/png;base64,{b64}" style="width:100%;border-radius:8px;">'


def build_html(stats, charts):
    now = datetime.now().strftime('%Y-%m-%d %H:%M')

    def kpi(label, value, unit=''):
        return f"""
        <div class="kpi">
            <div class="kpi-value">{value}<span class="kpi-unit">{unit}</span></div>
            <div class="kpi-label">{label}</div>
        </div>"""

    def section(title, content):
        return f"""
        <div class="section">
            <h2>{title}</h2>
            {content}
        </div>"""

    def card(content, cols=1):
        w = '100%' if cols == 1 else f'calc({100//cols}% - 12px)'
        return f'<div class="card" style="flex:0 0 {w};min-width:280px;">{content}</div>'

    def row(*cards):
        return f'<div class="row">{"".join(cards)}</div>'

    # Gender table
    gender_rows = ''.join(
        f'<tr><td>{k.title()}</td><td>{v}</td><td>{v/max(stats["total_visitors"],1)*100:.1f}%</td></tr>'
        for k, v in stats.get('gender_breakdown', {}).items()
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>IQMS Report — {now}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Segoe UI', Arial, sans-serif;
    background: {STYLE['bg']};
    color: {STYLE['text']};
    padding: 32px 24px;
  }}
  .header {{
    border-left: 5px solid {STYLE['accent']};
    padding-left: 20px;
    margin-bottom: 36px;
  }}
  .header h1 {{ font-size: 28px; font-weight: 700; }}
  .header p  {{ color: {STYLE['muted']}; margin-top: 6px; font-size: 14px; }}
  .section   {{ margin-bottom: 40px; }}
  .section h2 {{
    font-size: 17px;
    font-weight: 600;
    color: {STYLE['accent']};
    border-bottom: 1px solid {STYLE['grid']};
    padding-bottom: 8px;
    margin-bottom: 18px;
    text-transform: uppercase;
    letter-spacing: 1px;
  }}
  .row {{
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    margin-bottom: 16px;
  }}
  .card {{
    background: {STYLE['card']};
    border-radius: 10px;
    padding: 20px;
    border: 1px solid {STYLE['grid']};
  }}
  .card h3 {{
    font-size: 13px;
    color: {STYLE['muted']};
    margin-bottom: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}
  .kpi-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
    margin-bottom: 28px;
  }}
  .kpi {{
    background: {STYLE['card']};
    border: 1px solid {STYLE['grid']};
    border-radius: 10px;
    padding: 18px 24px;
    flex: 1;
    min-width: 140px;
    text-align: center;
  }}
  .kpi-value {{
    font-size: 32px;
    font-weight: 700;
    color: {STYLE['accent']};
  }}
  .kpi-unit  {{ font-size: 14px; color: {STYLE['muted']}; margin-left: 4px; }}
  .kpi-label {{ font-size: 12px; color: {STYLE['muted']}; margin-top: 6px; }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }}
  th {{
    text-align: left;
    padding: 8px 12px;
    color: {STYLE['muted']};
    border-bottom: 1px solid {STYLE['grid']};
    font-weight: 600;
    text-transform: uppercase;
    font-size: 11px;
  }}
  td {{
    padding: 8px 12px;
    border-bottom: 1px solid {STYLE['grid']};
  }}
  tr:last-child td {{ border-bottom: none; }}
  .badge {{
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    background: {STYLE['accent']};
    color: white;
  }}
  .footer {{
    margin-top: 48px;
    padding-top: 16px;
    border-top: 1px solid {STYLE['grid']};
    font-size: 12px;
    color: {STYLE['muted']};
  }}
</style>
</head>
<body>

<div class="header">
  <h1>Intelligent Queue Management System</h1>
  <p>Operational Report &nbsp;·&nbsp; Generated: {now} &nbsp;·&nbsp;
     Period: {stats.get('date_from','—')} → {stats.get('date_to','—')}</p>
</div>

<!-- KPIs -->
<div class="section">
  <h2>Executive Summary</h2>
  <div class="kpi-row">
    {kpi('Total Visitors', stats.get('total_visitors', 0))}
    {kpi('Days Monitored', stats.get('days_monitored', 0))}
    {kpi('Avg Visitors / Day', stats.get('avg_daily', 0))}
    {kpi('Peak Hour', f"{stats.get('peak_hour','—')}:00")}
    {kpi('Avg Age', stats.get('avg_age','—'), 'yrs')}
    {kpi('Checkout Events', stats.get('total_checkout_events', 0))}
    {kpi('Avg Service Time', stats.get('avg_service_time','—'), 's')}
  </div>
</div>

<!-- ENTRANCE TRAFFIC -->
<div class="section">
  <h2>Entrance Traffic Analysis</h2>

  {row(card(f'<h3>Daily Visitor Count</h3>{img_tag(charts.get("daily"))}'))}
  {row(card(f'<h3>Average Traffic by Hour</h3>{img_tag(charts.get("hourly_avg"))}'))}
  {row(card(f'<h3>Traffic Heatmap — Day × Hour</h3>{img_tag(charts.get("heatmap"))}'))}

  {row(
    card(f"""
      <h3>Gender Breakdown</h3>
      {img_tag(charts.get('gender'))}
      <table style="margin-top:14px;">
        <tr><th>Gender</th><th>Count</th><th>Share</th></tr>
        {gender_rows}
      </table>
    """),
    card(f'<h3>Age Distribution</h3>{img_tag(charts.get("age"))}')
  )}

  {row(card(f'<h3>Entrance Zone Dwell Time</h3>{img_tag(charts.get("dwell_entrance"))}'))}
</div>

<!-- CHECKOUT -->
<div class="section">
  <h2>Checkout Zone Analysis</h2>

  <div class="kpi-row">
    {kpi('Total Checkout Events', stats.get('total_checkout_events', 0))}
    {kpi('Avg Service Time', stats.get('avg_service_time','—'), 's')}
    {kpi('Max Service Time', stats.get('max_service_time','—'), 's')}
    {kpi('Busiest Hour', f"{stats.get('checkout_peak_hour','—')}:00")}
  </div>

  {row(card(f'<h3>Service Time Distribution</h3>{img_tag(charts.get("checkout_service"))}'))}
  {row(card(f'<h3>Avg Service Time by Hour</h3>{img_tag(charts.get("checkout_hourly"))}'))}
</div>

<!-- DATA QUALITY -->
<div class="section">
  <h2>Data Collection Status</h2>
  {row(card(f"""
    <table>
      <tr><th>Metric</th><th>Value</th></tr>
      <tr><td>Entrance records</td><td><span class="badge">{stats.get('total_visitors',0)}</span></td></tr>
      <tr><td>Checkout records</td><td><span class="badge">{stats.get('total_checkout_events',0)}</span></td></tr>
      <tr><td>Monitoring period</td><td>{stats.get('date_from','—')} → {stats.get('date_to','—')}</td></tr>
      <tr><td>Days with data</td><td>{stats.get('days_monitored',0)}</td></tr>
      <tr><td>Avg visitors / day</td><td>{stats.get('avg_daily',0)}</td></tr>
      <tr><td>Avg entrance dwell</td><td>{stats.get('avg_dwell_entrance','—')} s</td></tr>
      <tr><td>Prediction engine</td><td>Ready after 3–5 days of data</td></tr>
    </table>
  """))}
</div>

<div class="footer">
  IQMS · Intelligent Queue Management System · Auto-generated report · {now}
</div>

</body>
</html>"""
    return html


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  IQMS Report Generator")
    print("=" * 55)

    conn = connect()
    print("[Data] Loading entrance events...")
    ent = load_entrance(conn)
    print(f"[Data] Loaded {len(ent)} entrance records")

    print("[Data] Loading checkout events...")
    que = load_queue(conn)
    print(f"[Data] Loaded {len(que)} checkout records")
    conn.close()

    print("[Stats] Computing statistics...")
    stats = compute_stats(ent, que)

    print("[Charts] Generating charts...")
    charts = {}
    if not ent.empty:
        charts['daily']          = chart_daily_visitors(ent)
        charts['heatmap']        = chart_hourly_heatmap(ent)
        charts['hourly_avg']     = chart_hourly_avg(ent)
        charts['gender']         = chart_gender(ent)
        charts['age']            = chart_age(ent)
        charts['dwell_entrance'] = chart_dwell_entrance(ent)

    if not que.empty:
        charts['checkout_service'] = chart_checkout_service(que)
        charts['checkout_hourly']  = chart_checkout_hourly(que)

    print("[Report] Building HTML...")
    html = build_html(stats, charts)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(html)

    print("=" * 55)
    print(f"[Done] Report saved → {OUTPUT_FILE}")
    print(f"       Open it in any browser.")
    print("=" * 55)
    print(f"\n  Summary:")
    print(f"  · Visitors recorded : {stats.get('total_visitors', 0)}")
    print(f"  · Checkout events   : {stats.get('total_checkout_events', 0)}")
    print(f"  · Peak hour         : {stats.get('peak_hour','—')}:00")
    print(f"  · Avg daily visitors: {stats.get('avg_daily', 0)}")
    print(f"  · Avg service time  : {stats.get('avg_service_time','—')} s")


if __name__ == '__main__':
    main()
