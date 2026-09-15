from analyze import CircleGrid, readConfig
import json
import numpy as np

import matrix_utils
import os
from natsort import natsorted
from concurrent.futures import ProcessPoolExecutor

dataFolder = 'data/calib-150-times'

# selectedViews = range(25)
selectedViews = [0, 4, 20, 24]
# selectedViews = [0, 2, 4, 10, 12, 14, 20, 22, 24]


def process_single_folder(folder):
    # 将原来的逻辑封装进函数
    # 注意：analyzeFolder 必须是可序列化的（通常定义在顶层）
    return folder, analyzeFolder(folder)


def analyzeFolder(directory):
    analyzeConfig = readConfig("config/circle_loc.json")
    analyzeConfig['imageAndPoseDirectory'] = directory
    analyzeConfig['task'] = 'locMinReproject'
    pattern = CircleGrid(analyzeConfig)
    bTt = pattern.locMinReproject(returnBTt=True, selectedViews=selectedViews)
    vec = matrix_utils.matrixToTR(bTt)
    return np.array([vec])


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

toExport = {'selectedViews': list(selectedViews), 'data': all_results}
lenSelectedViews = len(selectedViews)

with open(f'result-json/minReproject-{lenSelectedViews}.json', 'w') as f:
    json.dump(toExport, f, indent=4)
