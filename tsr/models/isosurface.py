from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    # Official TripoSR uses torchmcubes.  Keep a CPU fallback for Windows
    # environments where compiling the extension is not possible.
    from torchmcubes import marching_cubes as _torchmcubes
except Exception:
    _torchmcubes = None


def _fallback_marching_cubes(level: torch.Tensor, threshold: float):
    from skimage import measure

    volume = level.detach().cpu().numpy().astype(np.float32)
    vertices, faces, _normals, _values = measure.marching_cubes(
        volume, level=float(threshold)
    )
    return (
        torch.from_numpy(vertices.astype(np.float32)),
        torch.from_numpy(faces.astype(np.int64)),
    )


class IsosurfaceHelper(nn.Module):
    points_range: Tuple[float, float] = (0, 1)

    @property
    def grid_vertices(self) -> torch.FloatTensor:
        raise NotImplementedError


class MarchingCubeHelper(IsosurfaceHelper):
    def __init__(self, resolution: int) -> None:
        super().__init__()
        self.resolution = resolution
        self.mc_func: Callable = _torchmcubes or _fallback_marching_cubes
        self._grid_vertices: Optional[torch.FloatTensor] = None

    @property
    def grid_vertices(self) -> torch.FloatTensor:
        if self._grid_vertices is None:
            # Keep the vertices on CPU so high resolutions do not duplicate the
            # full marching-cubes grid before the renderer chunks its queries.
            x, y, z = (
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
            )
            x, y, z = torch.meshgrid(x, y, z, indexing="ij")
            self._grid_vertices = torch.cat(
                [x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], dim=-1
            ).reshape(-1, 3)
        return self._grid_vertices

    def forward(
        self,
        level: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        level = -level.view(self.resolution, self.resolution, self.resolution)
        if _torchmcubes is not None:
            try:
                v_pos, t_pos_idx = self.mc_func(level.detach(), 0.0)
            except (AttributeError, RuntimeError):
                # Some Windows wheels expose the extension but only implement
                # CPU marching cubes.  Match the official CPU fallback.
                v_pos, t_pos_idx = self.mc_func(level.detach().cpu(), 0.0)
        else:
            v_pos, t_pos_idx = self.mc_func(level.detach().cpu(), 0.0)
        v_pos = v_pos[..., [2, 1, 0]]
        v_pos = v_pos / (self.resolution - 1.0)
        return v_pos.to(level.device), t_pos_idx.to(level.device)
