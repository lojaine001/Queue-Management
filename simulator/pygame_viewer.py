"""
Pygame viewer for the IQMS queue simulator.

Run with:
    python simulator/pygame_viewer.py
    python simulator/pygame_viewer.py --calibrate --days 30
"""

import argparse
import math
import os
import sys
from datetime import datetime, timedelta

import pygame

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulator.calibrate import calibrate
from simulator.db import SimDB
from simulator.engine import SimulatorEngine
from simulator.scenarios import SCENARIOS, get_scenario


WINDOW_W = 1320
WINDOW_H = 760
BG = (10, 15, 25)
SHOP_BG = (17, 55, 103)
SHOP_FILL = (18, 42, 75)
ENTRANCE_BG = (24, 112, 62)
LANE_BG = (127, 29, 29)
TEXT = (238, 241, 245)
SUBTEXT = (170, 180, 195)
WHITE = (245, 245, 245)
ORANGE = (249, 115, 22)
RED = (220, 38, 38)
MALE = (56, 189, 248)
FEMALE = (244, 114, 182)
UNKNOWN = (156, 163, 175)


class NullDB:
    def __init__(self, scenario_name: str):
        self.camera_id = f"SIM_PREVIEW_{scenario_name}"

    def insert_entrance(self, person):
        return None

    def update_dwell(self, person):
        return None

    def log_snapshot(self, timestamp, queue_count, avg_dwell, max_dwell, active_lanes):
        return None

    def log_service_event(self, person):
        return None

    def close(self):
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="Smooth Pygame viewer for the IQMS queue simulator.")
    parser.add_argument("scenario", nargs="?", default="normal_day", choices=list(SCENARIOS.keys()))
    parser.add_argument("--calibrate", action="store_true", help="Use REAL DB calibration before running.")
    parser.add_argument("--days", type=int, default=30, help="Days of REAL history for calibration.")
    parser.add_argument("--hours", type=int, default=8, help="Simulated duration in hours.")
    parser.add_argument("--speed", type=float, default=180.0, help="Simulated seconds per real second.")
    parser.add_argument("--fps", type=int, default=60, help="Render frames per second.")
    parser.add_argument("--persist", action="store_true", help="Write simulator rows to PostgreSQL.")
    return parser.parse_args()


def person_color(person, state: str):
    if state == "service":
        return RED
    if state == "queue":
        return ORANGE
    if person.gender == "male":
        return MALE
    if person.gender == "female":
        return FEMALE
    return UNKNOWN


def shop_position(person, current_time, shop_rect: pygame.Rect):
    if not person.shopping_end_eta or person.shopping_end_eta <= person.entry_time:
        progress = 0.0
    else:
        total = (person.shopping_end_eta - person.entry_time).total_seconds()
        elapsed = (current_time - person.entry_time).total_seconds()
        progress = min(1.0, max(0.0, elapsed / max(total, 1.0)))

    row = person.track_id % 5
    col_wobble = ((person.track_id % 3) - 1) * 12
    x = shop_rect.left + 35 + int((shop_rect.width - 80) * progress) + col_wobble
    y = shop_rect.top + 65 + row * ((shop_rect.height - 120) // 5)
    return x, y


def draw_diamond(screen, color, center, radius, border_color, border_width):
    x, y = center
    points = [(x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y)]
    pygame.draw.polygon(screen, color, points)
    pygame.draw.polygon(screen, border_color, points, border_width)


def draw_customer(screen, person, center, state: str):
    color = person_color(person, state)
    border = WHITE if person.is_group else (18, 24, 38)
    x, y = center

    if person.has_caddy:
        draw_diamond(screen, color, center, 10 if state == "service" else 8, border, 2)
    elif person.has_bag:
        size = 18 if state == "service" else 14
        rect = pygame.Rect(x - size // 2, y - size // 2, size, size)
        pygame.draw.rect(screen, color, rect, border_radius=4)
        pygame.draw.rect(screen, border, rect, 2, border_radius=4)
    else:
        radius = 10 if state == "service" else 8
        pygame.draw.circle(screen, color, center, radius)
        pygame.draw.circle(screen, border, center, radius, 2)


def draw_text(screen, font, text, pos, color=TEXT):
    surface = font.render(text, True, color)
    screen.blit(surface, pos)


def main():
    args = parse_args()
    config = get_scenario(args.scenario)
    if args.calibrate:
        print(f"[pygame_viewer] Applying calibration from last {args.days} days of REAL data")
        config.update(calibrate(days=args.days))

    start_time = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
    config["start_time"] = start_time
    config["end_time"] = start_time + timedelta(hours=args.hours)

    db = SimDB(f"PYGAME_{args.scenario}") if args.persist else NullDB(args.scenario)
    engine = SimulatorEngine(config, db)

    pygame.init()
    pygame.display.set_caption("IQMS Queue Simulator Viewer")
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    clock = pygame.time.Clock()
    title_font = pygame.font.SysFont("arial", 28, bold=True)
    body_font = pygame.font.SysFont("arial", 18)
    small_font = pygame.font.SysFont("arial", 15)

    floor_rect = pygame.Rect(35, 90, 1120, 610)
    entrance_rect = pygame.Rect(floor_rect.left, floor_rect.top, 120, floor_rect.height)
    shop_rect = pygame.Rect(entrance_rect.right + 10, floor_rect.top, 700, floor_rect.height)
    lane_area = pygame.Rect(shop_rect.right + 20, floor_rect.top + 20, 240, floor_rect.height - 40)

    running = True
    while running:
        dt = clock.tick(args.fps) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        if engine.has_pending_events() and engine.current_time < engine.end_time:
            target_time = engine.current_time + timedelta(seconds=dt * args.speed)
            engine.advance_until(target_time)

        screen.fill(BG)
        pygame.draw.rect(screen, (22, 29, 45), floor_rect, border_radius=12)
        pygame.draw.rect(screen, ENTRANCE_BG, entrance_rect, border_radius=10)
        pygame.draw.rect(screen, SHOP_FILL, shop_rect, border_radius=10)
        pygame.draw.rect(screen, SHOP_BG, shop_rect, 2, border_radius=10)
        pygame.draw.rect(screen, (70, 22, 22), lane_area, border_radius=10)

        draw_text(screen, title_font, "IQMS Queue Simulator Viewer", (35, 28))
        draw_text(screen, small_font, "ESC to quit", (1020, 36), SUBTEXT)
        draw_text(screen, body_font, "Entrance", (entrance_rect.left + 18, entrance_rect.top + 12))
        draw_text(screen, body_font, "Shop floor", (shop_rect.left + 18, shop_rect.top + 12))
        draw_text(screen, body_font, "Caisse lanes", (lane_area.left + 18, lane_area.top + 12))

        # Draw shoppers moving through the shop.
        for person in engine.people_in_shop[:90]:
            draw_customer(screen, person, shop_position(person, engine.current_time, shop_rect), "shopping")

        # Draw lanes, queue, and service.
        lane_gap = max(95, lane_area.height // max(len(engine.lanes), 1))
        for idx, lane in enumerate(engine.lanes):
            lane_y = lane_area.top + 55 + idx * lane_gap
            lane_rect = pygame.Rect(lane_area.left + 80, lane_y - 22, 110, 44)
            pygame.draw.rect(screen, LANE_BG, lane_rect, border_radius=8)
            pygame.draw.rect(screen, (240, 90, 90), lane_rect, 2, border_radius=8)
            draw_text(screen, small_font, f"L{idx + 1}", (lane_rect.left + 42, lane_rect.top + 12))

            if lane.current_person:
                draw_customer(screen, lane.current_person, (lane_rect.centerx, lane_rect.centery), "service")

            for q_idx, person in enumerate(lane.queue[:10]):
                q_x = lane_rect.left - 30 - q_idx * 24
                draw_customer(screen, person, (q_x, lane_rect.centery), "queue")

        # Metrics and legend.
        queue_count = sum(len(l.queue) + (1 if l.current_person else 0) for l in engine.lanes)
        info_x = 1185
        draw_text(screen, body_font, f"Sim time: {engine.current_time.strftime('%H:%M:%S')}", (info_x, 110))
        draw_text(screen, body_font, f"In shop: {len(engine.people_in_shop)}", (info_x, 145))
        draw_text(screen, body_font, f"In queue/service: {queue_count}", (info_x, 180))
        draw_text(screen, body_font, f"Completed: {len(engine.completed_people)}", (info_x, 215))
        draw_text(screen, body_font, f"Speed: {args.speed:.0f} sim sec / real sec", (info_x, 250))
        draw_text(screen, body_font, f"Persist DB: {'yes' if args.persist else 'no'}", (info_x, 285))
        draw_text(screen, body_font, "Legend", (info_x, 340))

        legend_y = 380
        samples = [
            ("Male", MALE, "circle", False),
            ("Female", FEMALE, "circle", False),
            ("Bag", UNKNOWN, "square", False),
            ("Caddy", UNKNOWN, "diamond", False),
            ("Group", UNKNOWN, "circle", True),
        ]
        for label, color, kind, grouped in samples:
            center = (info_x + 12, legend_y)
            sample_person = type("LegendPerson", (), {
                "gender": "male" if color == MALE else "female" if color == FEMALE else "unknown",
                "has_bag": kind == "square",
                "has_caddy": kind == "diamond",
                "is_group": grouped,
            })()
            draw_customer(screen, sample_person, center, "shopping")
            draw_text(screen, small_font, label, (info_x + 30, legend_y - 7))
            legend_y += 28

        if not engine.has_pending_events() or engine.current_time >= engine.end_time:
            draw_text(screen, body_font, "Simulation complete", (info_x, 560))

        pygame.display.flip()

    db.close()
    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
