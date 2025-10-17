#!/usr/bin/env python3
"""Interactive probe for the YCB Habitat adapter.

Run this script after downloading assets to confirm the scene loads,
observations stream out, and actions mutate the camera orbit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.env_ycb import EnvConfig, YCBHabitatAdapter
from src.utils.packets import PacketValidator

try:  # pragma: no cover - optional dependency
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency
    plt = None


def load_config(args: argparse.Namespace) -> EnvConfig:
    if args.config is None:
        payload: Dict[str, object] = {
            "scenario": "examiner",
            "backend": args.backend,
        }
    else:
        config_path = Path(args.config).expanduser().resolve()
        with config_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload.setdefault("backend", args.backend)
    if args.scene is not None:
        assets = payload.setdefault("assets", {})  # type: ignore[assignment]
        if isinstance(assets, dict):
            assets["examiner_scene"] = args.scene
    if args.objects:
        payload["objects"] = args.objects
    if args.gpu_device is not None:
        payload["gpu_device_id"] = args.gpu_device
    return EnvConfig.from_dict(payload)


def print_observation(obs: Dict[str, object]) -> None:
    meta = obs.get("global_meta", {}) if isinstance(obs, dict) else {}
    print(f"tick={meta.get('tick')} object={meta.get('object_id')}")
    columns = obs.get("columns", []) if isinstance(obs, dict) else []
    for column in columns:
        if not isinstance(column, dict):
            continue
        storage = column.get("patch", {}).get("storage") if isinstance(column.get("patch"), dict) else None
        channels = column.get("channels", [])
        print(
            f"  column={column.get('column_id')} storage={storage} modalities={channels} "
            f"pose=({column.get('egopose', {})})"
        )


def fetch_storage_arrays(adapter: YCBHabitatAdapter, storage_key: str) -> Dict[str, np.ndarray]:
    impl = getattr(adapter, "_impl", None)
    cache = getattr(impl, "_observation_cache", {}) if impl is not None else {}
    if not isinstance(cache, dict):
        return {}
    entry = cache.get(storage_key, {})
    if not isinstance(entry, dict):
        return {}
    return {name: np.asarray(value) for name, value in entry.items()}


def render_frames(adapter: YCBHabitatAdapter, observation: Dict[str, object]) -> None:
    if plt is None:
        print("matplotlib unavailable; install it to enable rendering.")
        return
    rendered = False
    columns = observation.get("columns", []) if isinstance(observation, dict) else []
    for column in columns:
        if not isinstance(column, dict):
            continue
        patch = column.get("patch")
        if not isinstance(patch, dict):
            continue
        storage = patch.get("storage")
        if not isinstance(storage, str):
            continue
        arrays = fetch_storage_arrays(adapter, storage)
        if not arrays:
            continue
        for name, array in arrays.items():
            fig = plt.figure(figsize=(4, 4))
            manager = getattr(fig.canvas, "manager", None)
            if manager is not None:
                try:
                    manager.set_window_title(f"{column.get('column_id')}:{name}")
                except Exception:
                    pass
            if name == "rgb":
                image = array
                if image.dtype != np.uint8:
                    image = np.clip(image, 0.0, 1.0)
                if image.shape[-1] == 4:
                    image = image[..., :3]
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                plt.imshow(image)
                plt.axis("off")
            else:
                plt.imshow(array, cmap="inferno")
                plt.colorbar()
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.05)
            rendered = True
    if not rendered:
        print("No cached frame data available for rendering (run with Habitat backend and step once).")


def build_action(tokens: Iterable[str]) -> Tuple[str, Dict[str, float]]:
    tokens = list(tokens)
    if not tokens:
        return "noop", {}
    cmd = tokens[0]
    if cmd == "move":
        dx = float(tokens[1]) if len(tokens) > 1 else 0.0
        dy = float(tokens[2]) if len(tokens) > 2 else 0.0
        return "move", {"dx": dx, "dy": dy}
    if cmd == "jump":
        u = float(tokens[1]) if len(tokens) > 1 else 0.5
        v = float(tokens[2]) if len(tokens) > 2 else 0.5
        return "jump_to", {"u": u, "v": v}
    if cmd in {"switch", "next"}:
        return "switch_object", {}
    if cmd == "reset":
        return "__RESET__", {}
    return "noop", {}


def interactive_loop(adapter: YCBHabitatAdapter) -> None:
    observation, pose, ctx = adapter.reset()
    print("Environment reset.")
    print_observation(observation)
    print(f"pose={pose} context_bits={len(ctx)}")
    print("Commands: step [move dx dy|jump u v|switch], show, stash <dir>, reset, quit")
    while True:
        try:
            raw = input("habitat> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw:
            raw = "step"
        tokens = raw.split()
        if tokens[0] in {"quit", "exit"}:
            break
        if tokens[0] == "show":
            print_observation(observation)
            render_frames(adapter, observation)
            continue
        if tokens[0] == "stash":
            if len(tokens) < 2:
                print("Usage: stash <directory>")
                continue
            target = Path(tokens[1]).expanduser().resolve()
            target.mkdir(parents=True, exist_ok=True)
            for column in observation.get("columns", []):  # type: ignore[assignment]
                if not isinstance(column, dict):
                    continue
                patch = column.get("patch")
                if not isinstance(patch, dict):
                    continue
                storage = patch.get("storage")
                if not isinstance(storage, str):
                    continue
                arrays = fetch_storage_arrays(adapter, storage)
                if not arrays:
                    print(f"No cached arrays for {storage}")
                    continue
                for name, array in arrays.items():
                    out_path = target / f"{storage.replace('/', '_')}_{name}.npy"
                    np.save(out_path, array)
                    print(f"Saved {name} -> {out_path}")
            continue
        action_type, params = build_action(tokens[1:]) if tokens[0] == "step" else build_action(tokens)
        if action_type == "__RESET__":
            observation, pose, ctx = adapter.reset()
            print("Environment reset.")
            print_observation(observation)
            print(f"pose={pose} context_bits={len(ctx)}")
            continue
        observation, pose, ctx = adapter.step({"action_type": action_type, "params": params})
        print_observation(observation)
        print(f"pose={pose} context_bits={len(ctx)}")


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactively probe the Habitat environment adapter.")
    parser.add_argument("--backend", choices=["stub", "habitat"], default="habitat")
    parser.add_argument("--config", help="Optional JSON config payload to override EnvConfig defaults.")
    parser.add_argument("--scene", help="Override the examiner scene GLB relative to assets.scenes.")
    parser.add_argument("--objects", nargs="*", help="Specify object ids to cycle through.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gpu-device", type=int, help="Override Habitat gpu_device_id (-1 for CPU).")
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)
    config = load_config(args)
    validator = PacketValidator()
    try:
        adapter = YCBHabitatAdapter(config, validator, rng)
    except Exception as exc:
        print(f"Failed to initialise adapter: {exc}", file=sys.stderr)
        return 2
    try:
        interactive_loop(adapter)
    finally:
        adapter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
