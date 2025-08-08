import mujoco
from mujoco import viewer
import numpy as np
from dynamixel_sdk import *  # Dynamixel SDK
import time

model = mujoco.MjModel.from_xml_path("SkyGrip_URDF.urdf")
data = mujoco.MjData(model)

with viewer.launch_passive(model, data) as v:
    while True:
        mujoco.mj_step(model, data)
        v.sync()
        time.sleep(0.01)