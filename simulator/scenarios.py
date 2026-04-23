from datetime import datetime, timedelta

DEFAULT_OPEN_HOUR = 8
DEFAULT_CLOSE_HOUR = 20

SCENARIOS = {
    "normal_day": {
        "num_lanes": 2,
        "avg_arrival_gap_seconds": 90,
        "prob_bag": 0.4,
        "prob_caddy": 0.1,
        "prob_bakery": 0.25,
        "prob_group": 0.15,
        "base_service_seconds": 45,
        "open_hour": DEFAULT_OPEN_HOUR,
        "close_hour": DEFAULT_CLOSE_HOUR,
    },
    "lunch_rush": {
        "num_lanes": 3,
        "avg_arrival_gap_seconds": 30,
        "prob_bag": 0.2,
        "prob_caddy": 0.05,
        "prob_bakery": 0.18,
        "prob_group": 0.3,
        "base_service_seconds": 30,
        "open_hour": DEFAULT_OPEN_HOUR,
        "close_hour": DEFAULT_CLOSE_HOUR,
    },
    "evening_rush": {
        "num_lanes": 4,
        "avg_arrival_gap_seconds": 45,
        "prob_bag": 0.6,
        "prob_caddy": 0.3,
        "prob_bakery": 0.12,
        "prob_group": 0.2,
        "base_service_seconds": 60,
        "open_hour": DEFAULT_OPEN_HOUR,
        "close_hour": DEFAULT_CLOSE_HOUR,
    }
}

def get_scenario(name: str):
    config = SCENARIOS.get(name, SCENARIOS["normal_day"]).copy()
    open_hour = int(config.get("open_hour", DEFAULT_OPEN_HOUR))
    close_hour = int(config.get("close_hour", DEFAULT_CLOSE_HOUR))
    now = datetime.now()
    config['open_hour'] = open_hour
    config['close_hour'] = close_hour
    config['start_time'] = now.replace(hour=open_hour, minute=0, second=0, microsecond=0)
    config['end_time'] = now.replace(hour=close_hour, minute=0, second=0, microsecond=0)
    return config
