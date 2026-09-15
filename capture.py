import numpy as np
import time
import json
import pyrealsense2 as rs2
import cv2
import sys
import csv
import os

from calculate_rotate import transform_xyz_tTc_to_bTc
from robot import Robot


# 初始化RealSense相机
class Camera:
    def __init__(self):
        self.pipe = rs2.pipeline()
        config = rs2.config()
        config.enable_stream(rs2.stream.color, 1920, 1080, rs2.format.bgr8, 30)
        self.pipe.start(config)

    def saveFrameTo(self, path):
        img = self.getRGBFrame()
        cv2.imwrite(path, img)

    def getRGBFrame(self):
        frames = self.pipe.wait_for_frames()
        img = np.asarray(frames.get_color_frame().get_data())
        return img

    def close(self):
        self.pipe.stop()


# 方法 2: 贪心算法
def greedyPath(allXYZ):
    allXYZ = np.array(allXYZ)
    assert allXYZ.shape[1] == 3
    resultAllXYZ = []
    resultAllXYZ.append(allXYZ[0])
    allXYZ = allXYZ[1:]
    while len(allXYZ) > 0:
        last = resultAllXYZ[-1]
        allDist = np.linalg.norm(allXYZ - last, axis=1)
        minIndex = np.argmin(allDist)
        resultAllXYZ.append(allXYZ[minIndex])
        allXYZ = np.delete(allXYZ, minIndex, axis=0)
    return resultAllXYZ


class MoveAndCapture:
    def __init__(self, configFile, locator, bTt):
        # 创建保存文件夹
        file_path = "./imr_take"
        self.file_path = file_path
        if not os.path.exists(file_path):
            os.mkdir(file_path)

        # 删除之前的文件
        for file in os.listdir(file_path):
            os.remove(f'{file_path}/{file}')

        # 打开文件准备保存位置数据
        fpose = open(f'{file_path}/pose.txt', 'w')
        self.fpose = fpose

        fpose.write('id,px,py,pz,rx,ry,rz\r\n')
        self.csvwriter = csv.writer(fpose)

        self.robot = Robot(locator)
        self.camera = None

        self.allDeltaPose = []
        self.readConfig(configFile)

        self.bTt = bTt

    def readConfig(self, file):
        with open(file, 'r') as f:
            config = json.load(f)
        self.config = config
        if 'matrix' not in config:
            self.config['matrix'] = 'A1'
        summary = f"Capture task: {config['task']}"
        if config['task'] != 'oneloc':
            summary += (
                f", x=[{config['minX']}, {config['maxX']}]"
                f", y=[{config['minY']}, {config['maxY']}]"
                f", z=[{config['minZ']}, {config['maxZ']}]"
            )
        print(summary)

    # 对于 calib 任务，顺序生成点
    # 对于 loc 任务，随机生成点
    # 对于 oneloc 任务，不需要生成点
    def generate_all_xyz(self, config):
        config = self.config

        task = config['task']

        if task == 'oneloc':
            return []

        minX = config['minX']
        maxX = config['maxX']
        minY = config['minY']
        maxY = config['maxY']
        minZ = config['minZ']
        maxZ = config['maxZ']
        if task == 'calib':
            pointNumX = config['calib']['pointNumX']
            pointNumY = config['calib']['pointNumY']
            pointNumZ = config['calib']['pointNumZ']
        elif task == 'loc' or task == 'loc-not-move':
            allNumber = config['loc']['allNumber']

        assert task in ['calib', 'loc', 'loc-not-move']
        allXYZ = []

        if task == 'calib':
            allX = np.linspace(minX, maxX, pointNumX)
            allY = np.linspace(minY, maxY, pointNumY)
            allZ = np.linspace(minZ, maxZ, pointNumZ)
            for z in allZ:
                for x in allX:
                    for y in allY:
                        allXYZ.append([x, y, z])
        else:
            np.random.seed(0)
            for i in range(allNumber):
                allXYZ.append([
                    np.random.uniform(minX, maxX),
                    np.random.uniform(minY, maxY),
                    np.random.uniform(minZ, maxZ),
                ])
        allXYZ = greedyPath(allXYZ)
        return allXYZ

    def getPoseN(self, n, interval):
        allPose = []
        for i in range(n):
            pose = self.robot.getPose()
            allPose.append(pose)
            time.sleep(interval)
        mean = list(np.mean(allPose, axis=0))
        return mean

    def savePhotoAndPose(self, index):
        time.sleep(0.5)  # 等待机械臂稳定
        self.camera = Camera()

        initial_state = self.getPoseN(33, .001)
        self.camera.saveFrameTo(f'{self.file_path}/{index}.jpg')
        self.camera.close()

        delta = np.zeros(6)
        self.allDeltaPose.append(delta)

        meanState = initial_state
        self.csvwriter.writerow([index] + list(meanState))

    def moveAndTakePhoto(self, nums, targetXYZ_tTc):
        point, new_orientation = transform_xyz_tTc_to_bTc(targetXYZ_tTc, self.bTt, self.config['matrix'])
        point /= 1000
        print(f"第{nums}个目标点：{point}, new_orientation:{new_orientation}")

        z = point[2]
        assert -.2 <= z <= 1

        # 执行机械臂运动
        pose = [point[0], point[1], point[2], new_orientation[0], new_orientation[1], new_orientation[2]]
        self.robot.movel(pose, 0.2, 0.1)

        # 拍照
        self.savePhotoAndPose(nums)

    def manualMoveAndTakePhoto(self, nums):
        while True:
            # 读取图片
            img = self.camera.getRGBFrame()
            if img is None:
                print('fail getRGBFrame')
                return None
            # resize to 640xauto
            width = 640
            height = int(img.shape[0] * (640 / img.shape[1]))
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)

            # 显示图片
            cv2.imshow('img', img)
            key = cv2.waitKey(1)
            if key == 32:  # space
                break
        print(f'save photo and pose   index: {nums}')
        self.savePhotoAndPose(nums)

    def moveAndCaptureAll(self):
        config = self.config
        task = config['task']

        if task == 'oneloc':
            self.savePhotoAndPose(0)
        elif task == 'calib' or task == 'loc':
            allXYZ = self.generate_all_xyz(config)
            for nums in range(len(allXYZ)):
                targetXYZ_tTc = allXYZ[nums]
                self.moveAndTakePhoto(nums, targetXYZ_tTc)
        elif task == 'loc-not-move':
            for nums in range(config['loc']['allNumber']):
                self.savePhotoAndPose(nums)

        mean = np.mean(self.allDeltaPose, axis=0)
        std = np.std(self.allDeltaPose, axis=0)

        with open(f'{self.file_path}/capture-log.txt', 'w') as f:
            f.write(f'mean delta: {mean}\n')
            f.write(f'std delta: {std}\n')

        self.fpose.close()
        self.robot.close()


if __name__ == '__main__':
    bTt = np.loadtxt('oneloc_bTt.txt', delimiter=',')
    assert len(sys.argv) == 2
    configFile = sys.argv[1]
    locator = None

    moveAndCapture = MoveAndCapture(configFile, locator, bTt)
    moveAndCapture.moveAndCaptureAll()
