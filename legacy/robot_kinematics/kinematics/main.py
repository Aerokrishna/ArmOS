import sys 
from pathlib import Path 

sys.path.insert(0, str(Path(__file__).resolve().parents[1])) 
from manyplot.frame_plot import plot_scene
import spatial_math as sm
import numpy as np

world = np.eye(4)

RWA = sm.rotX(np.radians(90))
RWB = sm.rotY(np.radians(90))

RAB = RWA.T @ RWB


print(sm.rot_to_XYZ(RAB))


T1 = sm.transformation(np.array([1,0,0]), RWA)
T2 = sm.transformation(np.array([0,0,1]), RWB)

plot_scene(frames = [T1, T2])

# print("Rotation Matrix R_XYZ_extrinsic:\n", R_XYZ_extrinsic)
# print("Rotation Matrix R_XYZ_intrinsic:\n", R_XYZ_intrinsic)
