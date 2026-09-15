import numpy as np
from scipy.spatial.transform import Rotation as R
import os

currentFilePath = os.path.abspath(__file__)
currentDir = os.path.dirname(currentFilePath)
os.chdir(currentDir)

gTc = np.loadtxt('gTc.txt', delimiter=',')

np.set_printoptions(precision=3, suppress=True)


def rotateMatrix(targetXYZ_tTc):
    normalized = targetXYZ_tTc / np.linalg.norm(targetXYZ_tTc)
    a = normalized[0]
    b = normalized[1]

    # {{p -> -((a b)/Sqrt[1 - b^2]), q -> Sqrt[1 - b^2], r -> a/Sqrt[1 - b^2], s -> -b Sqrt[(-1 + a^2 + b^2)/(-1 + b^2)], k -> -Sqrt[((-1 + a^2 + b^2)/(-1 + b^2))]}, {p -> (a b)/Sqrt[1 - b^2],
    # q -> -Sqrt[1 - b^2], r -> a/Sqrt[1 - b^2], s -> b Sqrt[(-1 + a^2 + b^2)/(-1 + b^2)], k -> -Sqrt[((-1 + a^2 + b^2)/(-1 + b^2))]}, {p -> -((a b)/Sqrt[1 - b^2]), q -> Sqrt[1 - b^2],
    # r -> -(a/Sqrt[1 - b^2]), s -> -b Sqrt[(-1 + a^2 + b^2)/(-1 + b^2)], k -> Sqrt[(-1 + a^2 + b^2)/(-1 + b^2)]}, {p -> (a b)/Sqrt[1 - b^2], q -> -Sqrt[1 - b^2], r -> -(a/Sqrt[1 - b^2]),
    # s -> b Sqrt[(-1 + a^2 + b^2)/(-1 + b^2)], k -> Sqrt[(-1 + a^2 + b^2)/(-1 + b^2)]}}
    p = -((a * b) / np.sqrt(1 - b**2))
    q = np.sqrt(1 - b**2)
    r = a / np.sqrt(1 - b**2)
    s = -b * np.sqrt((-1 + a**2 + b**2) / (-1 + b**2))
    k = -np.sqrt((-1 + a**2 + b**2) / (-1 + b**2))

    v1 = np.array([k, 0, r])
    v2 = np.array([p, q, s])
    v3 = np.array([-a, -b, -np.sqrt(1 - a**2 - b**2)])

    A1 = np.array([v1, v2, v3]).T
    A2 = np.array([v1, -v2, v3]).T
    A3 = np.array([-v1, v2, v3]).T
    A4 = np.array([-v1, -v2, v3]).T
    return A1, A2, A3, A4


def transform_xyz_tTc_to_bTc(targetXYZ_tTc, bTt, matrix, return_tTc=False, debug=False):
    A1, A2, A3, A4 = rotateMatrix(targetXYZ_tTc)

    if matrix == 'A1':
        A = A1
    elif matrix == 'A2':
        A = A2
    elif matrix == 'A3':
        A = A3
    elif matrix == 'A4':
        A = A4
    R_tTc = A

    target_tTc = np.eye(4)
    target_tTc[:3, 3] = targetXYZ_tTc
    target_tTc[:3, :3] = R_tTc

    if debug:
        print(f'Input bTt:\n{bTt}')
        print(f'Selected rotation matrix: {matrix}')
        print(f'Target tTc:\n{target_tTc}')
    if return_tTc:
        return target_tTc

    cTg = np.linalg.inv(gTc)
    target_bTg = bTt @ target_tTc @ cTg

    if debug:
        print(f'Target bTg:\n{target_bTg}')
    translationVector = target_bTg[:3, 3].flatten()
    rotateVector = R.from_matrix(target_bTg[:3, :3]).as_rotvec()
    return translationVector, rotateVector


if __name__ == '__main__':
    bTt = np.loadtxt('oneloc_bTt.txt', delimiter=',')
    targetXYZ_tTc = np.array([100, 100, 400])

    translationVector, rotateVector = transform_xyz_tTc_to_bTc(targetXYZ_tTc, bTt, 'A1', debug=True)
    print(f'Translation vector: {translationVector}')
    print(f'Rotation vector: {rotateVector}')
