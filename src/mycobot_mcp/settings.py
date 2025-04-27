import os

from pydantic import BaseModel


class Place(BaseModel):
    name: str
    description: str
    position: list[float]


class MyCobotSettings(BaseModel):
    urdf_path: str = os.path.join(os.path.dirname(__file__), "../../data/mycobot/mycobot.urdf")
    end_effector_name: str = "camera_flange"
    port: str = "/dev/ttyACM0"
    baud: int = 115200
    default_speed: int = 40
    default_z_speed: int = 20
    suction_pin: int = 5
    command_timeout: int = 5
    use_gravity_compensation: bool = False
    end_effector_height: float = 0.065  # pump head offset
    object_height: float = 0.01
    release_height: float = 0.05
    places: list[Place] = [
        Place(name="home", description="Home position", position=[0, 20, -130, 20, 0, 0]),
        Place(name="capture", description="Camera capture position", position=[0, 0, -30, -60, 0, -45]),
        Place(name="drop", description="Position to drop object", position=[-45, 20, -130, 20, 0, 0]),
    ]


class MyCobotMCPSettings(BaseModel):
    camera_id: int = 0
    camera_parameter_path: str = os.path.join(os.path.dirname(__file__), "../../camera_calibration/mtx_dist.npz")
    mycobot_settings: MyCobotSettings = MyCobotSettings()
