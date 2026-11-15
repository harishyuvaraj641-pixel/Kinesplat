"""Small shared data classes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Camera:
    """An OpenCV-convention pinhole camera.

    ``R`` / ``t`` are the *camera-to-world* transform, so a world point ``X``
    maps to camera coordinates as ``R.T @ (X - t)``.  Camera axes are
    x-right, y-down, z-forward.
    """

    name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    R: np.ndarray
    t: np.ndarray
    image_path: str = ""
    depth_path: str = ""

    def __post_init__(self) -> None:
        self.R = np.asarray(self.R, dtype=np.float64).reshape(3, 3)
        self.t = np.asarray(self.t, dtype=np.float64).reshape(3)

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def position(self) -> np.ndarray:
        return self.t.copy()

    @property
    def world_to_camera(self) -> tuple[np.ndarray, np.ndarray]:
        R = self.R.T
        t = -R @ self.t
        return R, t

    def scaled(self, new_width: int, new_height: int) -> "Camera":
        sx = new_width / float(self.width)
        sy = new_height / float(self.height)
        return Camera(
            name=self.name,
            width=new_width,
            height=new_height,
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=self.cx * sx,
            cy=self.cy * sy,
            R=self.R.copy(),
            t=self.t.copy(),
            image_path=self.image_path,
            depth_path=self.depth_path,
        )

    def look_direction(self) -> np.ndarray:
        return self.R[:, 2].copy()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "fx": float(self.fx),
            "fy": float(self.fy),
            "cx": float(self.cx),
            "cy": float(self.cy),
            "R": self.R.tolist(),
            "t": self.t.tolist(),
            "image_path": self.image_path,
            "depth_path": self.depth_path,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Camera":
        return cls(
            name=data["name"],
            width=int(data["width"]),
            height=int(data["height"]),
            fx=float(data["fx"]),
            fy=float(data["fy"]),
            cx=float(data["cx"]),
            cy=float(data["cy"]),
            R=np.asarray(data["R"], dtype=np.float64),
            t=np.asarray(data["t"], dtype=np.float64),
            image_path=data.get("image_path", ""),
            depth_path=data.get("depth_path", ""),
        )


@dataclass
class SparsePoint:
    xyz: np.ndarray
    rgb: np.ndarray


@dataclass
class Scene:
    cameras: list[Camera]
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    colors: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    source: str = "unknown"
    scale: float = 1.0

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if len(self.points):
            lo = self.points.min(axis=0).astype(np.float64)
            hi = self.points.max(axis=0).astype(np.float64)
        else:
            pos = np.array([c.position for c in self.cameras])
            lo, hi = pos.min(axis=0), pos.max(axis=0)
        return lo, hi

    def scaled(self, scale: float) -> "Scene":
        return Scene(
            cameras=[c.scaled(c.width, c.height) for c in self.cameras],
            points=self.points.copy(),
            colors=self.colors.copy(),
            source=self.source,
            scale=self.scale,
        )
