"""
对前 60 组多视角数据，使用最小化重投影误差联合优化一个 bTt 位姿。
优化变量：bTt（6 自由度：旋转向量 3 + 平移向量 3）
残差：所有 60 组 × v 个视角 × p 个关键点的重投影误差（2 × 60 × v × p 维）
"""
import numpy as np
import cv2 as cv
from scipy.optimize import least_squares
import os
import sys
import time
import json
from natsort import natsorted

from analyze import (
    CircleGrid,
    readConfig,
    readIntrinsic,
    parse_pose_txt,
    read_bTg,
    read_all_bTg,
    getAllImage,
)
from matrix_utils import *

np.set_printoptions(precision=6, suppress=True)
np.set_printoptions(linewidth=120)

# ============ 配置 ============
dataFolder = 'data/calib-150-times'
numFolders = 60

# selectedViews = range(25)
selectedViews = [0, 4, 20, 24]  # 4 个视角
# selectedViews = [0, 2, 4, 10, 12, 14, 20, 22, 24]  # 9 个视角

# =============================

# 读取共享的相机内参和手眼标定结果
gTc, camera_matrix, dist_coeffs = readIntrinsic()
camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64)
gTc = np.asarray(gTc, dtype=np.float64)

# 创建 CircleGrid 实例用于圆心检测
config = readConfig("config/circle_loc.json")
config['imageAndPoseDirectory'] = dataFolder + '/repeatCalib_0/'  # 临时占位
config['task'] = 'locMinReproject'
pattern = CircleGrid(config)

# 获取目标点（世界坐标）
object_points = np.asarray(pattern.getObjectPoints(), dtype=np.float64)
object_points_cv = object_points.reshape(-1, 1, 3)

# 收集所有文件夹的数据：与 runMultiView 中 train 部分一致（取前 60 个文件夹）
allFolders = natsorted(os.listdir(dataFolder))
allFolders = [f'{dataFolder}/{f}/' for f in allFolders]
folders = allFolders[:numFolders]  # 与 runMultiView allFolders[:60] 一致
print(f'共 {len(folders)} 个文件夹（从第 0 个开始）')
print(f'每文件夹选取视角: {list(selectedViews)}')

all_bTc = []  # 每个视角的 base->camera 变换
all_image_pts = []  # 每个视角的观测图像点
all_folder_indices = []  # 记录每个视角属于哪个文件夹（调试用）

startTime = time.time()
totalViews = 0

for folderIdx, folder in enumerate(folders):
    allImageFile = getAllImage(folder)
    all_bTg = read_all_bTg(folder, allImageFile)

    for viewIdx in selectedViews:
        if viewIdx >= len(allImageFile):
            continue
        image_file = allImageFile[viewIdx]
        try:
            # 使用 CircleGrid 的圆心检测 + 细化
            centers = pattern._detectCircleGridRefine(image_file)
            centers = np.asarray(centers, dtype=np.float64)

            bTg = all_bTg[viewIdx]
            bTc = bTg @ gTc

            all_bTc.append(bTc)
            all_image_pts.append(centers)
            all_folder_indices.append(folderIdx)
            totalViews += 1
        except Exception as e:
            print(f'[警告] 文件夹 {folder} 视角 {viewIdx} 检测失败: {e}')
            continue

detectTime = time.time() - startTime
print(f'成功检测 {totalViews} 个视角，耗时 {detectTime:.2f}s')

all_bTc = np.array(all_bTc)
all_image_pts = np.array(all_image_pts)
print(f'bTc shape: {all_bTc.shape}, image_pts shape: {all_image_pts.shape}')

# ============ 优化 ============
# 先用每个视角独立 PnP 的结果做平均作为初值
all_bTt_init = []
for centers, bTc in zip(all_image_pts, all_bTc):
    _, rvec, tvec = cv.solvePnP(object_points, centers, camera_matrix, dist_coeffs)
    cTt = rtToMatrix(rvec.flatten(), tvec.flatten())
    all_bTt_init.append(bTc @ cTt)

# 四元数平均求初值
init_bTt = pattern.meanMatrix(all_bTt_init)
init_rvec, _ = cv.Rodrigues(init_bTt[:3, :3])
x0 = np.zeros(6, dtype=np.float64)
x0[:3] = init_rvec.flatten()
x0[3:] = init_bTt[:3, 3]
print(f'初值 bTt:\n{init_bTt}')


def residuals(x):
    """重投影误差：对所有文件夹的所有视角，计算投影误差"""
    bTt = rtToMatrix(x[:3], x[3:6])
    res = []
    for i in range(totalViews):
        bTc = all_bTc[i]
        cTt = np.linalg.inv(bTc) @ bTt
        rvec_ct, _ = cv.Rodrigues(cTt[:3, :3])
        tvec_ct = cTt[:3, 3].reshape(3, 1)

        imgpts, _ = cv.projectPoints(object_points_cv, rvec_ct, tvec_ct, camera_matrix, dist_coeffs)
        imgpts = imgpts.reshape(-1, 2)

        obs = all_image_pts[i]
        valid = ~np.isnan(obs).any(axis=1)
        diff = imgpts[valid] - obs[valid]
        res.extend(diff.ravel())
    return np.asarray(res, dtype=np.float64)


# 检查初始残差
res0 = residuals(x0)
n_residuals = res0.size
print(f'残差维度: {n_residuals}')
print(f'初始残差范数: {np.linalg.norm(res0):.4f}')
print(f'初始 RMSE (px): {np.sqrt(np.mean(res0**2)):.4f}')

# 优化
method = 'lm' if n_residuals >= x0.size else 'trf'
optStart = time.time()
result = least_squares(residuals, x0, method=method, xtol=1e-12, ftol=1e-12, gtol=1e-12)
optTime = time.time() - optStart

opt_x = result.x
bTt_opt = rtToMatrix(opt_x[:3], opt_x[3:6])

print('\n============ 优化结果 ============')
print(f'状态: {result.status}  ({result.message})')
print(f'优化耗时: {optTime:.2f}s')
print(f'最终残差范数: {np.linalg.norm(result.fun):.4f}')
print(f'最终 RMSE (px): {np.sqrt(np.mean(result.fun**2)):.4f}')
print(f'函数评估次数: {result.nfev}')
print(f'雅可比评估次数: {result.njev}')
print('\n优化后的 bTt (base -> target，单位: mm):')
print(bTt_opt)

# 转成 tq 格式打印
tq_opt = matrixToTQ(bTt_opt)
print(f'\nt,q 格式: {tq_opt}')

# 转成米制打印
bTt_opt_m = bTt_opt.copy()
bTt_opt_m[:3, 3] /= 1000
print('\nbTt (单位: m):')
print(bTt_opt_m)

totalTime = time.time() - startTime
print(f'\n总耗时: {totalTime:.2f}s (检测 {detectTime:.2f}s + 优化 {optTime:.2f}s)')

# ============ 导出 JSON ============
bTt_tr = matrixToTR(bTt_opt).tolist()
output_data = {
    'selectedViews': list(selectedViews),
    'numFolders': numFolders,
    'folders': folders,
    'bTt_tr': bTt_tr,
    'optimize_status': result.status,
    'optimize_message': result.message,
    'final_rmse_px': float(np.sqrt(np.mean(result.fun**2))),
    'n_residuals': int(n_residuals),
}
output_path = f'result-json/minReproject-60-{len(selectedViews)}.json'
with open(output_path, 'w') as f:
    json.dump(output_data, f, indent=4)
print(f'\n结果已导出至: {output_path}')
