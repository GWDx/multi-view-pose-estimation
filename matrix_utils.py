import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2 as cv


# 旋转向量、平移向量 -> 矩阵
def rtToMatrix(rotvec, tvec):
    rotMatrix = cv.Rodrigues(rotvec)[0]
    mat = np.eye(4)
    mat[:3, :3] = rotMatrix
    mat[:3, 3] = tvec
    return mat


# 四元数 -> 旋转向量
def quaternionToRotvec(q):
    return R.from_quat(q).as_rotvec()


# 方向向量 -> 旋转
def vecToMatrix(vec):
    return R.from_rotvec(vec).as_matrix()


# 旋转矩阵 -> 四元数
def matrixToQuat(mat):
    rotation_mat = R.from_matrix(mat)
    return rotation_mat.as_quat()


# 旋转矩阵 -> 四元数
def quatToMatrix(quat):
    rot = R.from_quat(quat)  # 顺序为 (x, y, z, w)
    return rot.as_matrix()


# 旋转矩阵、平移向量 -> 矩阵
def rotateMatrixTranslationVectorToMatrix(rot, t):
    mat = np.eye(4)
    mat[:3, :3] = rot
    mat[:3, 3] = t
    return mat


def fast_matrix_to_quat(m):
    """
    将 3x3 旋转矩阵转换为单位四元数 [qx, qy, qz, qw]
    采用数值稳定的分情况处理方法。
    """
    trace = np.trace(m)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (m[2, 1] - m[1, 2]) * s
        qy = (m[0, 2] - m[2, 0]) * s
        qz = (m[1, 0] - m[0, 1]) * s
    else:
        # 如果 trace <= 0，找到主对角线上最大的元素
        if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            qw = (m[2, 1] - m[1, 2]) / s
            qx = 0.25 * s
            qy = (m[0, 1] + m[1, 0]) / s
            qz = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            qw = (m[0, 2] - m[2, 0]) / s
            qx = (m[0, 1] + m[1, 0]) / s
            qy = 0.25 * s
            qz = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            qw = (m[1, 0] - m[0, 1]) / s
            qx = (m[0, 2] + m[2, 0]) / s
            qy = (m[1, 2] + m[2, 1]) / s
            qz = 0.25 * s

    return np.array([qx, qy, qz, qw])


# 矩阵 -> 平移向量、四元数连接起来
def matrixToTQ(mat):
    rot = mat[:3, :3]
    t = mat[:3, 3]
    quaternion = fast_matrix_to_quat(rot)
    result = np.concatenate((t, quaternion))
    return result


# 平移向量、四元数 -> 矩阵
def tqToMatrix(tq):
    t = tq[:3]
    q = tq[3:]
    rot = quatToMatrix(q)
    return rotateMatrixTranslationVectorToMatrix(rot, t)


def matrixToTR(mat):
    rot = mat[:3, :3]
    t = mat[:3, 3]
    rotvec = R.from_matrix(rot).as_rotvec()
    result = np.concatenate((t, rotvec))
    return result


def trToMatrix(tr):
    t = tr[:3]
    rvec = tr[3:]
    rot = R.from_rotvec(rvec).as_matrix()
    return rotateMatrixTranslationVectorToMatrix(rot, t)
