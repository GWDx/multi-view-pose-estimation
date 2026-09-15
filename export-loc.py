from analyze import CircleGrid, readConfig
import json
import numpy as np

import matrix_utils
import os
from natsort import natsorted
from concurrent.futures import ProcessPoolExecutor

dataFolder = 'data/calib-150-times'


def process_single_folder(folder):
    # 将原来的逻辑封装进函数
    # 注意：analyzeFolder 必须是可序列化的（通常定义在顶层）
    return folder, analyzeFolder(folder)


def analyzeFolder(directory):
    analyzeConfig = readConfig("config/circle_loc.json")
    analyzeConfig['imageAndPoseDirectory'] = directory
    pattern = CircleGrid(analyzeConfig)
    all_bTt = pattern.loc(returnAllBTt=True)
    all_vec = [matrix_utils.matrixToTR(bTt) for bTt in all_bTt]
    return np.array(all_vec)


folders = natsorted(os.listdir(dataFolder))
folders = [f'{dataFolder}/{f}/' for f in folders]

all_results = {}

folders_to_process = [f for f in folders if f not in all_results]
print(f"Processing {len(folders_to_process)} folders")

# max_workers 默认是 CPU 核心数
with ProcessPoolExecutor(max_workers=10) as executor:
    # map 会保持顺序返回结果
    results = list(executor.map(process_single_folder, folders_to_process))

# 更新到最终字典
for folder, res in results:
    # numpy to list
    all_results[folder] = res.tolist()
print(f'length: {len(all_results)}')

with open('result-json/allBTt.json', 'w') as f:
    json.dump(all_results, f, indent=4)
