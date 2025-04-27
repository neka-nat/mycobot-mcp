import numpy as np
from pydantic import BaseModel, ConfigDict

class FlatPixelToWorld(BaseModel):
    matrix: np.ndarray
    distortion: np.ndarray
    flip_x: int = -1
    flip_y: int = -1

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_camera_parameters_path(cls, path: str) -> "FlatPixelToWorld":
        with np.load(path) as data:
            mtx = data["mtx"]
            dist = data["dist"]
        return cls(matrix=mtx, distortion=dist)

    def uv_to_xy(self, u: float, v: float, height: float) -> tuple[float, float]:
        dx_pix = (u - self.matrix[0, 2]) * self.flip_x
        dy_pix = (v - self.matrix[1, 2]) * self.flip_y
        x = dx_pix * height / self.matrix[0, 0]
        y = dy_pix * height / self.matrix[1, 1]
        return (float(x), float(y))
