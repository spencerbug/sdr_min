#!/usr/bin/env python3
"""Emit a minimal black-void GLB scene for the Examiner scenario."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a minimal black void GLB scene.")
    parser.add_argument(
        "--dest",
        default="assets/scenes/examiner/void_black.glb",
        help="Output path for the GLB file.",
    )
    parser.add_argument(
        "--extent",
        type=float,
        default=5.0,
        help="Half-extent of the surrounding cube (default: 5 meters).",
    )
    return parser.parse_args(argv)


def build_cube(extent: float) -> tuple[np.ndarray, np.ndarray]:
    e = extent
    vertices = np.array(
        [
            [-e, -e, -e],
            [e, -e, -e],
            [e, e, -e],
            [-e, e, -e],
            [-e, -e, e],
            [e, -e, e],
            [e, e, e],
            [-e, e, e],
        ],
        dtype=np.float32,
    )
    indices = np.array(
        [
            0, 2, 1, 0, 3, 2,
            4, 5, 6, 4, 6, 7,
            0, 1, 5, 0, 5, 4,
            2, 3, 7, 2, 7, 6,
            0, 4, 7, 0, 7, 3,
            1, 2, 6, 1, 6, 5,
        ],
        dtype=np.uint16,
    )
    return vertices, indices


def write_glb(path: Path, extent: float) -> None:
    try:
        from pygltflib import (  # type: ignore[import]
            Accessor,
            Asset,
            Buffer,
            BufferView,
            GLTF2,
            Material,
            Mesh,
            Node,
            PbrMetallicRoughness,
            Primitive,
            Scene,
        )
    except ImportError as exc:  # pragma: no cover - runtime guard only
        raise SystemExit("pygltflib is required; install with pip install pygltflib") from exc

    vertices, indices = build_cube(extent)
    buffer_data = bytearray(vertices.nbytes + indices.nbytes)
    buffer_data[: vertices.nbytes] = vertices.tobytes()
    buffer_data[vertices.nbytes :] = indices.tobytes()

    buffer = Buffer(byteLength=len(buffer_data))
    buffer_view_positions = BufferView(buffer=0, byteOffset=0, byteLength=vertices.nbytes, target=34962)
    buffer_view_indices = BufferView(buffer=0, byteOffset=vertices.nbytes, byteLength=indices.nbytes, target=34963)

    accessor_positions = Accessor(
        bufferView=0,
        byteOffset=0,
        componentType=5126,  # FLOAT
        count=len(vertices),
        type="VEC3",
        min=vertices.min(axis=0).astype(float).tolist(),
        max=vertices.max(axis=0).astype(float).tolist(),
    )
    accessor_indices = Accessor(
        bufferView=1,
        byteOffset=0,
        componentType=5123,  # UNSIGNED_SHORT
        count=len(indices),
        type="SCALAR",
    )

    primitive = Primitive(attributes={"POSITION": 0}, indices=1, material=0, mode=4)
    material = Material(
        name="void",
        pbrMetallicRoughness=PbrMetallicRoughness(
            baseColorFactor=[0.0, 0.0, 0.0, 1.0], metallicFactor=0.0, roughnessFactor=1.0
        ),
    )
    mesh = Mesh(primitives=[primitive], name="void_cube")
    node = Node(mesh=0, name="void_root")
    scene = Scene(nodes=[0], name="void_scene")

    gltf = GLTF2(
        asset=Asset(version="2.0"),
        scenes=[scene],
        scene=0,
        nodes=[node],
        meshes=[mesh],
        materials=[material],
        buffers=[buffer],
        bufferViews=[buffer_view_positions, buffer_view_indices],
        accessors=[accessor_positions, accessor_indices],
    )

    gltf.set_binary_blob(bytes(buffer_data))
    gltf.save(path)


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    path = Path(args.dest)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_glb(path, args.extent)
    return path


if __name__ == "__main__":
    main()
