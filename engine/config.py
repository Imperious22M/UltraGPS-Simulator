"""
Configuration files.

  config/config.toml   receiver positions and calibration: what the positioning
                       library believes (UltraGPS-Control's format)
  config/sensors.toml  the simulated hardware's truth per receiver: tick
                       response, facing direction and, optionally, where it
                       really stands
  config/world.toml    the room: walls and obstacles sound bounces off
"""

from __future__ import annotations

import os

import numpy as np

from engine.hardware import Receiver
from engine.world import Obstacle, World, default_walls
from ultragps_position import load_config

_REPO           = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG  = os.path.join(_REPO, "config", "config.toml")
DEFAULT_SENSORS = os.path.join(_REPO, "config", "sensors.toml")
DEFAULT_WORLD   = os.path.join(_REPO, "config", "world.toml")


def believed_positions(config_path: str) -> np.ndarray:
    receivers = sorted(load_config(config_path)["receivers"], key=lambda r: r["id"])
    return np.array([r["position"] for r in receivers], dtype=float)


def load_receivers(config_path: str, sensors_path: str) -> list[Receiver]:
    """Receivers in id order: true response, facing and position from
    sensors.toml; position falls back to config.toml's, facing to the arena centre."""
    believed = believed_positions(config_path)
    truth    = {int(r["id"]): r for r in load_config(sensors_path)["receivers"]}
    centre   = (believed.min(axis=0) + believed.max(axis=0)) / 2
    out = []
    for i, pos in enumerate(believed):
        t      = truth[i]
        true   = np.array(t.get("position", pos), dtype=float)
        to_c   = centre - true
        facing = float(t.get("facing_deg", np.degrees(np.arctan2(to_c[1], to_c[0]))))
        out.append(Receiver(i, true, facing, float(t["slope"]), float(t["intercept"])))
    return out


def load_world(world_path: str | None, coords) -> World:
    """The room in world.toml, or walls 50 cm outside the receivers if there is none."""
    if not world_path or not os.path.exists(world_path):
        return World(default_walls(coords, 50.0), 0.6, [])
    cfg   = load_config(world_path)
    walls = [tuple(map(float, p)) for p in cfg.get("walls", [])]
    obs   = [Obstacle([tuple(map(float, p)) for p in o["points"]],
                      float(o.get("reflectivity", 0.7)), str(o.get("name", "")))
             for o in cfg.get("obstacles", [])]
    return World(walls, float(cfg.get("wall_reflectivity", 0.6)), obs)


def save_world(world_path: str, world: World) -> None:
    def pts(points):
        return "[" + ", ".join(f"[{x:.1f}, {y:.1f}]" for x, y in points) + "]"
    lines = [
        "# The room the simulated sound travels in (centimetres, same axes as config.toml).",
        "#",
        "# walls: one closed polygon round the arena; remove it for an open field.",
        "# reflectivity: share of the sound's amplitude a surface reflects (0..1);",
        "# hard walls about 0.6-0.9, cloth or foam much less.",
        "# obstacles: convex polygons that block and reflect sound.",
        "",
        f"wall_reflectivity = {world.wall_reflectivity:.2f}",
        f"walls = {pts(world.walls)}" if world.walls else "walls = []",
    ]
    for o in world.obstacles:
        lines += ["", "[[obstacles]]", f'name = "{o.name}"',
                  f"reflectivity = {o.reflectivity:.2f}", f"points = {pts(o.points)}"]
    with open(world_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
