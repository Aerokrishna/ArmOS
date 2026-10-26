import numpy as np

def rotX(theta):
    return np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]], dtype=np.float64)

def rotY(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)],[0, 1, 0],[-np.sin(theta), 0, np.cos(theta)]], dtype=np.float64)

def rotZ(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],[np.sin(theta),  np.cos(theta), 0],[0, 0, 1]], dtype=np.float64)

def translate(p: np.ndarray):
    T = np.eye(4)
    T[:3, 3] = p
    return T

def transformation(p: np.ndarray, R: np.ndarray):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T

def rot_to_XYZ(R):
    """
    Convert rotation matrix to extrinsic XYZ Euler angles.
    """

    if R.shape != (3, 3):
        raise ValueError("Input must be 3x3 rotation matrix")

    R = np.asarray(R, dtype=float)

    ry = np.arcsin(-R[2, 0])

    if np.abs(np.cos(ry)) > 1e-8:

        rx = np.arctan2(R[2, 1], R[2, 2])
        rz = np.arctan2(R[1, 0], R[0, 0])

    else:
        rx = 0.0
        rz = np.arctan2(-R[0, 1], R[1, 1])

    return np.array([rx, ry, rz])

def XYZ_to_rot(rx, ry, rz):
    """
    Convert extrinsic XYZ Euler angles to rotation matrix.
    """

    ca, sa = np.cos(rx), np.sin(rx)
    cb, sb = np.cos(ry), np.sin(ry)
    cg, sg = np.cos(rz), np.sin(rz)

    Rx = rotX(rx)
    Ry = rotY(ry)
    Rz = rotZ(rz)
    # Extrinsic XYZ
    R = Rz @ Ry @ Rx

    return R

def transform_vec_to_target(T_world_child: np.ndarray, p_child: np.ndarray) -> np.ndarray:
    p_child_h = np.append(p_child, 1.0)      
    p_world_h = T_world_child @ p_child_h    
    return p_world_h[:3]                     

def so3_vec(v: np.ndarray):
    """
    Convert vector to skew-symmetric matrix
    """
    return np.array([[0, -v[2], v[1]], 
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def adjoint_transform(T):

    R = T[:3, :3]
    p = T[:3, 3]

    adjT = np.zeros((6, 6))

    adjT[:3, :3] = R
    adjT[3:, 3:] = R
    adjT[3:, :3] = so3_vec(p) @ R

    return adjT

def transform_vel(vec, T):
    """
    Transform wrench or twist using adjoint transform
    """

    adjT = adjoint_transform(T)

    vec_parent = adjT @ vec

    return vec_parent

# given vector, make it a frame with z aligned to it
def dirvec_to_R(dir_vec):
    dir_vec = np.asarray(dir_vec, dtype=float)

    if np.linalg.norm(dir_vec) < 1e-8:
        raise ValueError("Direction vector cannot be zero")

    # New Z axis
    z_new = dir_vec / np.linalg.norm(dir_vec)

    # World X axis (reference for minimal rotation)
    x_ref = np.array([1.0, 0.0, 0.0])

    # Project reference X onto plane orthogonal to Z
    x_proj = x_ref - np.dot(x_ref, z_new) * z_new

    # Handle singularity when dir_vec ≈ X axis
    if np.linalg.norm(x_proj) < 1e-6:
        y_ref = np.array([0.0, 1.0, 0.0])
        x_proj = np.cross(y_ref, z_new)

    x_new = x_proj / np.linalg.norm(x_proj)

    # Compute Y axis
    y_new = np.cross(z_new, x_new)
    y_new /= np.linalg.norm(y_new)

    # Recompute X to ensure perfect orthogonality
    x_new = np.cross(y_new, z_new)

    # Rotation matrix
    R_new = np.column_stack((x_new, y_new, z_new))

    return R_new

# given a direction vector and a theta, compute rotation matrix
def integrate_rotation(dir_vec, theta):

    w = np.asarray(dir_vec, dtype=float)
    w = w / np.linalg.norm(w)

    w_hat = so3_vec(w)

    I = np.eye(3)

    R = I + np.sin(theta) * w_hat + (1 - np.cos(theta)) * (w_hat @ w_hat)

    return R

def skew_to_vector(M):
    """Skew matrix -> vector"""
    return np.array([M[2,1],M[0,2],M[1,0]])


def rotation_log_error(R_des, R):
    """
    Log map rotation error on SO(3)

    Returns axis-angle error vector (theta * axis)
    """

    R_e = R_des @ R.T

    cos_theta = (np.trace(R_e) - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    theta = np.arccos(cos_theta)

    if theta < 1e-6:
        return np.zeros(3)

    log_R = (theta / (2 * np.sin(theta))) * (R_e - R_e.T)

    e_R = skew_to_vector(log_R)

    return e_R


def xyz_extrinsic_to_quat(rx, ry, rz):
    """
    Extrinsic XYZ rotation -> quaternion [w,x,y,z]
    """

    cx = np.cos(rx/2)
    sx = np.sin(rx/2)

    cy = np.cos(ry/2)
    sy = np.sin(ry/2)

    cz = np.cos(rz/2)
    sz = np.sin(rz/2)

    w = cx*cy*cz + sx*sy*sz
    x = sx*cy*cz - cx*sy*sz
    y = cx*sy*cz + sx*cy*sz
    z = cx*cy*sz - sx*sy*cz

    return np.array([w,x,y,z])

def quat_to_xyz_extrinsic(q):
    """
    quaternion [w,x,y,z] -> extrinsic XYZ angles
    """

    w,x,y,z = q

    # roll (x)
    rx = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    # pitch (y)
    ry = np.arcsin(2*(w*y - z*x))
    # yaw (z)
    rz = np.arctan2(2*(w*z + x*y),1 - 2*(y*y + z*z))

    return np.array([rx,ry,rz])

def normalize(q):
    return q/np.linalg.norm(q)

def quat_multiply(q1, q2):

    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return np.array([w,x,y,z])

def quat_conjugate(q):
    w,x,y,z = q
    return np.array([w,-x,-y,-z])

def quat_log_error(q_t: np.ndarray, q_c: np.ndarray) -> np.ndarray:

    q_c_inv = quat_conjugate(q_c)
    q_e = quat_multiply(q_t, q_c_inv)

    # ensure shortest rotation
    if q_e[0] < 0.0:
        q_e *= -1.0

    ang = 2.0 * np.arccos(np.clip(q_e[0], -1.0, 1.0))

    if ang < 1e-6:
        return np.zeros(3)

    axis = q_e[1:] / np.sin(ang/2.0)

    return ang * axis


print(so3_vec(np.array([1, 2, 3])))