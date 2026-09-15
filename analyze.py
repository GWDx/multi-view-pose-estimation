import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
import os
import json
import sys

from matrix_utils import *
from statsmodels.stats.stattools import durbin_watson

np.set_printoptions(precision=6, suppress=True)
np.set_printoptions(linewidth=120)


def distort_points_manual(pts, K, dist):
    """
    手工将无畸变的图像像素点加上针孔+徕卡式（OpenCV常用）畸变模型。
    pts: (N,2) 像素坐标（undistorted）
    K: (3,3) 相机内参矩阵
    dist: 1D array_like, 长度常见为 4,5,8 (k1,k2,p1,p2[,k3[,k4,k5,k6...]])
    返回: (N,2) 畸变后的像素坐标
    """
    pts = np.asarray(pts, dtype=np.float64)
    dist = np.asarray(dist, dtype=np.float64).ravel()
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (pts[:, 0] - cx) / fx
    y = (pts[:, 1] - cy) / fy
    r2 = x * x + y * y
    # radial terms
    k1 = dist[0] if dist.size > 0 else 0.0
    k2 = dist[1] if dist.size > 1 else 0.0
    p1 = dist[2] if dist.size > 2 else 0.0  # 注意：OpenCV 的顺序常见为 (k1,k2,p1,p2,k3,...)
    p2 = dist[3] if dist.size > 3 else 0.0
    k3 = dist[4] if dist.size > 4 else 0.0
    radial = 1.0 + k1 * r2 + k2 * (r2**2) + k3 * (r2**3)
    # tangential
    x_t = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    y_t = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y

    x_dist = x * radial + x_t
    y_dist = y * radial + y_t

    # 如果有更多的径向系数（k4,k5,k6），OpenCV 在更高版本里会有更多项，
    # 这里为简单起见没有额外实现。如果你有 k4,k5,k6，可以按需要添加。
    u_dist = x_dist * fx + cx
    v_dist = y_dist * fy + cy
    return np.column_stack((u_dist, v_dist))


def durbinWatsonTest(data):
    dw = durbin_watson(data)
    print(f'Durbin-Watson statistic: {dw:.6f}')


def shapiro_test(data):
    stat, p = shapiro(data)
    print()
    print(f"Shapiro-Wilk Test: W = {stat:.4f}, p-value = {p:.4f}")
    if p > 0.05:
        print("结论: 数据服从正态分布 (无法拒绝原假设)")
    else:
        print("结论: 数据不服从正态分布 (拒绝原假设)")


def readConfig(file):
    with open(file, 'r') as f:
        config = json.load(f)
    return config


def read_gTc():
    # 获取相机内参
    gTc = np.loadtxt('gTc.txt', delimiter=',')
    return gTc


def readCameraMatrix():
    camera_matrix = np.loadtxt('camera_matrix.txt', delimiter=',')
    dist_coeffs = np.loadtxt('dist_coeffs.txt', delimiter=',')
    return camera_matrix, dist_coeffs


# 读取相机内参
def readIntrinsic():
    gTc = read_gTc()
    camera_matrix, dist_coeffs = readCameraMatrix()
    return gTc, camera_matrix, dist_coeffs


def parse_pose_txt(txt_path):
    fpose = open(txt_path, 'r')
    ftxt = fpose.readlines()
    if ftxt[0].startswith('id,'):
        ftxt.pop(0)
    poses = {}
    for i in range(0, len(ftxt)):
        fline = ftxt[i]
        data = [float(x) for x in fline.split(',')]
        xyz = data[1:4]
        xyz = [1000 * x for x in xyz]
        rxyz = data[4:]
        r = R.from_rotvec(rxyz)

        quaternion = r.as_quat()
        poses[int(data[0])] = {'xyz': xyz, 'quaternion': quaternion}
    return poses


def getAllImage(imageAndPoseDirectory):
    files = os.listdir(imageAndPoseDirectory)
    imageFiles = [file for file in files if file.endswith('.jpg')]
    imageFiles.sort(key=lambda x: int(x.split('.')[0]))
    imageFiles = [imageAndPoseDirectory + file for file in imageFiles]
    return imageFiles


def read_bTg(imageAndPoseDirectory, imageIndex):
    allPose = parse_pose_txt(imageAndPoseDirectory + '/pose.txt')

    pose = allPose[imageIndex]
    tVec = pose['xyz']
    qVec = pose['quaternion']
    rVec = quaternionToRotvec(qVec)
    bTg = rtToMatrix(rVec, tVec)
    return bTg


def read_all_bTg(imageAndPoseDirectory, allImageFile):
    all_bTg = []
    for imageFile in allImageFile:
        index = int(imageFile.split('/')[-1].split('.')[0])
        temp_bTg = read_bTg(imageAndPoseDirectory, index)
        all_bTg.append(temp_bTg)
    return all_bTg


def showUndistortedImage(image_file, camera_matrix, dist_coeffs):
    image = cv.imread(image_file)
    image_undistorted = cv.undistort(image, camera_matrix, dist_coeffs)
    cv.imshow('undistorted', image_undistorted)
    cv.waitKey(0)


def generateRectangle(minX, maxX, minY, maxY):
    return np.array([[minX, minY, 0], [maxX, minY, 0], [maxX, maxY, 0], [minX, maxY, 0]])


# 输入是 n*3 的矩阵，输出 4 个点的 2d 坐标
def movePointsToRectangleCenter(object_points):
    allX = object_points[:, 0]
    allY = object_points[:, 1]
    allZ = object_points[:, 2]

    # 保证 z 轴都是 0
    assert np.all(allZ == 0)

    # 计算最小外接矩形
    minX = np.min(allX)
    maxX = np.max(allX)
    minY = np.min(allY)
    maxY = np.max(allY)

    center = np.array([(minX + maxX) / 2, (minY + maxY) / 2, 0])
    newObjectPoints = object_points - center
    rectanglePoints = generateRectangle(minX, maxX, minY, maxY)
    return newObjectPoints, rectanglePoints, center


class Pattern:
    def __init__(self, config):
        task = config['task']
        self.task = task
        self.type = config['type']
        self.imageAndPoseDirectory = config['imageAndPoseDirectory']
        self.poseRelativeFile = config['poseRelativeFile']
        self.calibResultFile = config['calibResultFile']
        self.maxImageLength = config['maxImageLength']

        if task == 'oneloc':
            self.imageRelativeFile = config['oneloc']['imageRelativeFile']

        if task in ('loc', 'oneloc', 'locMinReproject'):
            gTc, camera_matrix, dist_coeffs = readIntrinsic()
            self.gTc = gTc
            self.camera_matrix = camera_matrix
            self.dist_coeffs = dist_coeffs

    def calculate_bTt_std(self, imageLength, all_bTt):
        all_bTt = np.array(all_bTt)
        # 第奇数个的变换乘上第偶数个的逆变换，计算差值相对于 gt=0 的误差
        diffOddEven = [np.dot(all_bTt[i], np.linalg.inv(all_bTt[i + 1])) for i in range(0, len(all_bTt) - 1, 2)]

        diffOddEven = np.array(diffOddEven)

        # Durbin-Watson 检验 diffOddEven 的 x
        durbinWatsonTest(diffOddEven[:, 0, 3])
        durbinWatsonTest(diffOddEven[:, 1, 3])
        durbinWatsonTest(diffOddEven[:, 2, 3])

        durbinWatsonTest(all_bTt[:, 0, 3] - np.mean(all_bTt[:, 0, 3]))
        durbinWatsonTest(all_bTt[:, 1, 3] - np.mean(all_bTt[:, 1, 3]))
        durbinWatsonTest(all_bTt[:, 2, 3] - np.mean(all_bTt[:, 2, 3]))

        all_tq_diff = [matrixToTQ(diff) for diff in diffOddEven]
        all_tq_diff = np.array(all_tq_diff)

        all_tq_bTt = [matrixToTQ(bTt) for bTt in all_bTt]
        all_tq_bTt = np.array(all_tq_bTt)

        mean_diff = np.mean(all_tq_diff, axis=0)
        print(f'Mean relative pose difference: {mean_diff}')

        # 计算精度
        std_bTt = np.std(all_tq_bTt, axis=0)
        print(f'Pose standard deviation: {std_bTt}')

        std_diff = np.std(all_tq_diff, axis=0)
        print(f'Relative pose standard deviation: {std_diff}')

        # 计算每个旋转的角度
        all_tq_diff_rotvec = np.array([R.from_quat(tq[3:]).as_rotvec() for tq in all_tq_diff])
        all_tq_diff_angle = np.linalg.norm(all_tq_diff_rotvec, axis=1)
        # 转成角度
        all_tq_diff_angle = np.degrees(all_tq_diff_angle)
        print(f'Mean relative rotation angle (degrees): {np.mean(all_tq_diff_angle):.6f}')

        std_err_diff = std_diff / np.sqrt(2 * (len(all_tq_diff) - 1))
        print(f'Standard error of relative pose: {std_err_diff}')

        target = np.array([0, 0, 0, 0, 0, 0, 1])
        newRMSE = np.sqrt(np.mean(np.square(all_tq_diff - target), axis=0))

        print('A^-1 B 相对于不变换，每个轴的 RMSE 是：', newRMSE)

        np.savetxt('RMSE_xyz.txt', newRMSE[:3], fmt='%.6f', delimiter=',')

    def calculate_cTt(self, object_points, image_points, camera_matrix, dist_coeffs):
        assert len(object_points) == len(image_points)
        assert image_points.shape[1] == 2
        assert object_points.shape[1] == 3

        _, pnp_rvecs, pnp_tvecs = cv.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
        pnp_rvecs = pnp_rvecs.flatten()
        pnp_tvecs = pnp_tvecs.flatten()
        rot_vec = np.array([np.pi, 0, 0])
        pnp_rvecs = (R.from_rotvec(pnp_rvecs) * R.from_rotvec(rot_vec)).as_rotvec()
        cTt = rtToMatrix(pnp_rvecs, pnp_tvecs)
        return cTt

    def meanMatrix(self, part_bTt):
        # 转成四元数，平均，归一化以后再转回旋转矩阵
        part_quaternions = [matrixToTQ(bTt) for bTt in part_bTt]
        part_quaternions = np.array(part_quaternions)

        mean_quaternion = np.mean(part_quaternions, axis=0)
        mean_t, mean_q = mean_quaternion[:3], mean_quaternion[3:]
        mean_q = mean_q / np.linalg.norm(mean_q)
        mean_quaternion = np.append(mean_t, mean_q)
        mean_matrix = tqToMatrix(mean_quaternion)
        return mean_matrix

    def calculateAllBTt(self, all_cTt, allImageFile):
        # 准备 bTg, gTc
        all_bTg = read_all_bTg(self.imageAndPoseDirectory, allImageFile)
        gTc = read_gTc()

        # 使用矩阵乘法计算 bTt
        all_data_bTt = []
        for bTg, cTt in zip(all_bTg, all_cTt):
            bTt = bTg @ gTc @ cTt
            all_data_bTt.append(bTt)
        all_data_bTt = np.array(all_data_bTt)
        return all_data_bTt

    def analyzeBTt(self, all_data_bTt, imageLength, allImageFile):
        mean_matrix = self.meanMatrix(all_data_bTt)
        self.calculate_bTt_std(imageLength, all_data_bTt)
        print('base 到 target 的齐次矩阵是：')
        print(mean_matrix)
        np.savetxt('bTt.txt', mean_matrix, fmt='%.6f', delimiter=',')
        mean_matrix[:3, 3] /= 1000
        print(mean_matrix)
        np.savetxt(self.imageAndPoseDirectory + 'bTt_mean_matrix.txt', mean_matrix, fmt='%.6f', delimiter=',')
        return mean_matrix

    def calculateOneloc(self, object_points, image_points):
        # 计算相机坐标系下 target 的位姿
        cTt = self.calculate_cTt(object_points, image_points, self.camera_matrix, self.dist_coeffs)
        gTc = self.gTc

        # 计算 target 坐标系下的相机位姿
        imageIndex = int(self.imageRelativeFile.split('.')[0])
        bTg = read_bTg(self.imageAndPoseDirectory, imageIndex)
        tTc = np.linalg.inv(cTt)

        bTc = np.dot(bTg, gTc)
        bTt = np.dot(bTc, cTt)

        print(f'Estimated bTt:\n{bTt}')
        np.savetxt('oneloc_bTt.txt', bTt, fmt='%.6f', delimiter=',')
        return bTt

    def calculateCalib(self, imageLength, allImageFile, rvecs, tvecs):
        # 准备 bTg, cTt
        all_bTg = read_all_bTg(self.imageAndPoseDirectory, allImageFile)

        all_cTt = []
        for i in range(imageLength):
            temp_cTt = rtToMatrix(rvecs[i], tvecs[i].flatten())
            all_cTt.append(temp_cTt)

        R_bTg = np.array([bTg[:3, :3] for bTg in all_bTg])
        t_bTg = np.array([bTg[:3, 3] for bTg in all_bTg])
        R_cTt = np.array([cTt[:3, :3] for cTt in all_cTt])
        t_cTt = np.array([cTt[:3, 3] for cTt in all_cTt])

        # 手眼标定，获得 gTc
        R_gTc, t_gTc = cv.calibrateHandEye(R_bTg, t_bTg, R_cTt, t_cTt, method=cv.CALIB_HAND_EYE_DANIILIDIS)
        gTc = rotateMatrixTranslationVectorToMatrix(R_gTc, t_gTc.flatten())
        print(f'Estimated hand-eye transform gTc:\n{gTc}')

        # tT0 is move (60,20,0)
        rotvec0 = np.array([np.pi, 0, 0])
        tvec0 = np.array([60, 20, 0])
        tT0 = rtToMatrix(rotvec0, tvec0)

        # 保存结果
        all_bT0 = []
        all_bTt = []
        for bTg, cTt in zip(all_bTg, all_cTt):
            bT0 = bTg @ gTc @ cTt @ tT0
            all_bT0.append(bT0)
            bTt = bTg @ gTc @ cTt
            all_bTt.append(bTt)

        avg_bT0 = np.mean(all_bT0, axis=0)
        avg_tq_bT0 = matrixToTQ(avg_bT0)

        print(f'Average reference pose: {avg_tq_bT0}')

        self.calculate_bTt_std(imageLength, all_bTt)
        # 不严格区分 id=0 的 aruco 码的坐标系、target 坐标系
        return gTc


class Charuco(Pattern):
    def __init__(self, config):
        # 父类初始化
        super(Charuco, self).__init__(config)
        assert self.type == 'charuco'

        self.ARUCO_DICT = config['charuco']['ARUCO_DICT']
        self.SQUARE_LENGTH = config['charuco']['SQUARE_LENGTH']
        self.MARKER_LENGTH = config['charuco']['MARKER_LENGTH']
        self.SQUARES_VERTICALLY = config['charuco']['SQUARES_VERTICALLY']
        self.SQUARES_HORIZONTALLY = config['charuco']['SQUARES_HORIZONTALLY']
        self.selectedAruco = config['charuco']['selectedAruco']

        # cv.aruco.getPredefinedDictionary
        arucoDictID = eval('cv.aruco.' + self.ARUCO_DICT)
        self.arucoDict = cv.aruco.getPredefinedDictionary(arucoDictID)

        self.board = cv.aruco.CharucoBoard((self.SQUARES_VERTICALLY, self.SQUARES_HORIZONTALLY), self.SQUARE_LENGTH,
                                           self.MARKER_LENGTH, self.arucoDict)

    # 获取图片中所有 ArUco 码的位置
    def _getAllArucoPosition(self, image):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        parameters = cv.aruco.DetectorParameters()
        aruco_detector = cv.aruco.ArucoDetector(self.arucoDict)
        marker_corners, marker_ids, rejected_candidates = aruco_detector.detectMarkers(gray)
        if marker_ids is None:
            raise Exception("No ArUco markers detected.")
        marker_corners = np.array(marker_corners)
        marker_ids = np.array(marker_ids)

        assert marker_corners.shape[1] == 1
        assert marker_corners.shape[2] == 4
        assert marker_corners.shape[3] == 2
        assert marker_ids.shape[1] == 1
        assert marker_ids.shape[0] == marker_corners.shape[0]

        # reshape to (n, 4, 2) and (n)
        marker_corners = marker_corners.reshape(-1, 4, 2)
        marker_ids = marker_ids.reshape(-1)
        return marker_corners, marker_ids

    # 过滤出指定 ArUco 码的位置（oneloc 任务专用）
    def _filterArucoPosition(self, marker_corners, marker_ids, selectedAruco):
        selectedMarkerCorners = []
        selectedMarkerIds = []
        for arucoID, marker_corner in zip(marker_ids, marker_corners):
            if arucoID in selectedAruco:
                selectedMarkerCorners.append(marker_corner)
                selectedMarkerIds.append(arucoID)
        selectedMarkerCorners = np.array(selectedMarkerCorners)
        selectedMarkerIds = np.array(selectedMarkerIds)
        return selectedMarkerCorners, selectedMarkerIds

    # 输入 (n, 4, 2) 的角点位置，返回 flatten 后的结果
    def _getImagePoints(self, marker_corners):
        image_points = np.vstack(marker_corners)
        return image_points

    # 获得角点在世界坐标系中的位置，输入所有 ArUco 码的四个角点和 ID，返回 flatten 后的结果
    def _getObjectPoints(self, selectedMarkerIds):
        squareCount = self.SQUARES_VERTICALLY * self.SQUARES_HORIZONTALLY // 2
        aruco_cnt = 0
        aruco_object_points = np.empty((squareCount, 4, 3), dtype=np.float32)

        halfMarkerLength = self.MARKER_LENGTH / 2
        center_object_point = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]) * halfMarkerLength

        for i in range(0, self.SQUARES_HORIZONTALLY):
            for j in range(0, self.SQUARES_VERTICALLY):
                if (i + j) % 2 == 0:
                    continue

                aruco_object_points[aruco_cnt] = center_object_point
                aruco_object_points[aruco_cnt, :, 1] += float(i) * self.SQUARE_LENGTH
                aruco_object_points[aruco_cnt, :, 0] += float(j - 1) * self.SQUARE_LENGTH
                aruco_cnt += 1

        object_points = np.array([aruco_object_points[markerID] for markerID in selectedMarkerIds.flatten()])

        assert object_points.shape[1] == 4
        assert object_points.shape[2] == 3
        object_points = object_points.reshape(-1, 3)

        resultObjectPoints = object_points
        return resultObjectPoints

    def oneloc(self):
        image_path = self.imageAndPoseDirectory + self.imageRelativeFile
        image = cv.imread(image_path)

        # 所有 ArUco 码的角点位置
        marker_corners, marker_ids = self._getAllArucoPosition(image)

        # 过滤出指定 ArUco 码角点位置
        selectedAruco = self.selectedAruco
        selectedMarkerCorners, selectedMarkerIds = self._filterArucoPosition(marker_corners, marker_ids, selectedAruco)

        # 获得 target 坐标系下的 aruco 码角点位置
        object_points = self._getObjectPoints(selectedMarkerIds)
        newObjectPoints, rectanglePoints, center = movePointsToRectangleCenter(object_points)
        image_points = self._getImagePoints(selectedMarkerCorners)

        # 计算 target 坐标系下的相机位姿
        result = self.calculateOneloc(newObjectPoints, image_points)
        return result

    def loc(self):
        allImageFile = getAllImage(self.imageAndPoseDirectory)
        imageLength = len(allImageFile)
        np.random.shuffle(allImageFile)
        all_cTt = []

        # 从每一张图片中获取 cTt
        for image_file in allImageFile:
            image = cv.imread(image_file)

            # 所有 ArUco 码的角点位置
            marker_corners, marker_ids = self._getAllArucoPosition(image)

            # 过滤出指定 ArUco 码角点位置
            selectedAruco = self.selectedAruco
            selectedMarkerCorners, selectedMarkerIds = self._filterArucoPosition(marker_corners, marker_ids,
                                                                                 selectedAruco)

            # 获得 target 坐标系下的 aruco 码角点位置
            object_points = self._getObjectPoints(selectedMarkerIds)
            image_points = self._getImagePoints(selectedMarkerCorners)
            newObjectPoints, rectanglePoints, center = movePointsToRectangleCenter(object_points)

            # cornerSubPix
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            winSize = (5, 5)
            zeroZone = (-1, -1)
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 40, 0.001)
            image_points = cv.cornerSubPix(gray, image_points, winSize, zeroZone, criteria)

            # 计算 target 坐标系下的相机位姿
            cTt = self.calculate_cTt(newObjectPoints, image_points, camera_matrix, dist_coeffs)

            all_cTt.append(cTt)

        return self.calculateLoc(all_cTt, imageLength, allImageFile)

    def calib(self):
        allImageFile = getAllImage(self.imageAndPoseDirectory)
        np.random.shuffle(allImageFile)
        imageLength = len(allImageFile)

        all_charuco_corners = []
        all_charuco_ids = []

        for image_file in allImageFile:
            image = cv.imread(image_file)

            charucoDetector = cv.aruco.CharucoDetector(self.board)
            charuco_corners, charuco_ids, marker_corners, marker_ids = charucoDetector.detectBoard(image)

            all_charuco_corners.append(charuco_corners)
            all_charuco_ids.append(charuco_ids)

        # 相机标定
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.aruco.calibrateCameraCharuco(
            all_charuco_corners, all_charuco_ids, self.board, image.shape[:2][::-1], None, None)
        print(f'Camera matrix:\n{camera_matrix}')
        print(f'Distortion coefficients:\n{dist_coeffs}')

        # write to camera_matrix.txt
        np.savetxt('camera_matrix.txt', camera_matrix, fmt='%.6f', delimiter=',')
        np.savetxt('dist_coeffs.txt', dist_coeffs, fmt='%.6f', delimiter=',')
        gTc = self.calculateCalib(imageLength, allImageFile, rvecs, tvecs)
        np.savetxt('gTc.txt', gTc, fmt='%.6f', delimiter=',')
        return gTc


class Aruco(Pattern):
    def __init__(self, config):
        super(Aruco, self).__init__(config)
        assert self.type == 'aruco'
        assert self.task != 'calib'

        self.ARUCO_DICT = config['aruco']['ARUCO_DICT']
        self.MARKER_LENGTH = config['aruco']['MARKER_LENGTH']

        arucoDictID = eval('cv.aruco.' + self.ARUCO_DICT)
        self.arucoDict = cv.aruco.getPredefinedDictionary(arucoDictID)

    # 获取图片中所有 ArUco 码的位置
    def _getArucoPosition(self, image):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        parameters = cv.aruco.DetectorParameters()
        aruco_detector = cv.aruco.ArucoDetector(self.arucoDict)
        marker_corners, marker_ids, rejected_candidates = aruco_detector.detectMarkers(gray)
        if marker_ids is None:
            raise Exception("No ArUco markers detected.")

        assert len(marker_corners) == 1
        oneMarkerCorners = marker_corners[0]
        oneMarkerIds = marker_ids[0]

        return oneMarkerCorners, oneMarkerIds

    def oneloc(self):
        image_path = self.imageAndPoseDirectory + self.imageRelativeFile
        image = cv.imread(image_path)

        # 所有 ArUco 码的角点位置
        marker_corners, marker_ids = self._getArucoPosition(image)

        halfMarkerLength = self.MARKER_LENGTH / 2
        center_object_point = generateRectangle(-halfMarkerLength, halfMarkerLength, -halfMarkerLength,
                                                halfMarkerLength)
        # 获得 target 坐标系下的 aruco 码角点位置
        object_points = center_object_point
        image_points = marker_corners[0]

        # 计算 target 坐标系下的相机位姿
        result = self.calculateOneloc(object_points, image_points)
        return result

    def loc(self):
        allImageFile = getAllImage(self.imageAndPoseDirectory)
        imageLength = len(allImageFile)
        np.random.shuffle(allImageFile)
        all_cTt = []

        # 从每一张图片中获取 cTt
        for image_file in allImageFile:
            image = cv.imread(image_file)

            # 所有 ArUco 码的角点位置
            marker_corners, marker_ids = self._getArucoPosition(image)

            # 获得 target 坐标系下的 aruco 码角点位置
            halfMarkerLength = self.MARKER_LENGTH / 2
            center_object_point = generateRectangle(-halfMarkerLength, halfMarkerLength, -halfMarkerLength,
                                                    halfMarkerLength)
            object_points = center_object_point
            image_points = marker_corners[0]

            camera_matrix = self.camera_matrix
            dist_coeffs = self.dist_coeffs
            # 计算 target 坐标系下的相机位姿
            cTt = self.calculate_cTt(object_points, image_points, camera_matrix, dist_coeffs)

            all_cTt.append(cTt)

        return self.calculateLoc(all_cTt, imageLength, allImageFile)


class CircleGrid(Pattern):
    def __init__(self, config):
        super(CircleGrid, self).__init__(config)
        assert self.type == 'circleGrid'

        self.circleDistance = config['circleGrid']['circleDistance']
        self.circleRadius = config['circleGrid']['circleRadius']
        self.circlePerRow = config['circleGrid']['circlePerRow']

    def _detectCircleGrid(self, image):
        if isinstance(image, str):
            image = cv.imread(image)
        # 所有 ArUco 码的角点位置
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        patternSize = (self.circlePerRow, self.circlePerRow)

        # 调整亮度
        alpha = 1.0
        beta = 50
        gray = cv.convertScaleAbs(gray, alpha=alpha, beta=beta)

        params = cv.SimpleBlobDetector.Params()
        detector = cv.SimpleBlobDetector.create(params)
        retval, centers = cv.findCirclesGrid(gray,
                                             patternSize,
                                             flags=cv.CALIB_CB_SYMMETRIC_GRID | cv.CALIB_CB_CLUSTERING,
                                             blobDetector=detector)

        assert centers.shape == (self.circlePerRow * self.circlePerRow, 1, 2)
        centersGrid = centers.reshape(self.circlePerRow, self.circlePerRow, 2)

        allCenterGrid = []

        for i in range(4):
            allCenterGrid.append(centersGrid)
            centersGrid = np.rot90(centersGrid)

        def getGray(image, centerGrid):
            center = np.mean(centerGrid.reshape(self.circlePerRow * self.circlePerRow, 2), axis=0)
            fourCorner = [centerGrid[0][0], centerGrid[-1][0], centerGrid[-1][-1], centerGrid[0][-1]]
            fourCornerExtend = [center + 5 / 4 * (corner - center) for corner in fourCorner]
            # 仿射变换到正方形
            squareSize = 300
            destPoints = np.array([[0, 0], [0, squareSize], [squareSize, squareSize], [squareSize, 0]], np.float32)
            M = cv.getPerspectiveTransform(np.array(fourCornerExtend, np.float32), destPoints)
            image = cv.warpPerspective(image, M, (squareSize, squareSize))

            # crop
            cornerSquareSize = squareSize // 16
            meanGray = np.mean(image[:cornerSquareSize, :cornerSquareSize])
            return meanGray

        allGray = [getGray(image, centerGrid) for centerGrid in allCenterGrid]

        # 选择最小灰度的 index 的 centerGrid
        minIndex = np.argmin(allGray)
        allGray.sort()
        if allGray[1] - allGray[0] <= 10:
            raise Exception('too small')
        centersGrid = allCenterGrid[minIndex]
        centers = centersGrid.reshape(self.circlePerRow * self.circlePerRow, 1, 2)
        assert len(centers) == self.circlePerRow * self.circlePerRow
        centers = centers.reshape(self.circlePerRow * self.circlePerRow, 2)
        return centers

    def getObjectPoints(self):
        circleGridObjectPositions = np.zeros((self.circlePerRow * self.circlePerRow, 3), np.float32)
        for i in range(self.circlePerRow):
            for j in range(self.circlePerRow):
                circleGridObjectPositions[i * self.circlePerRow +
                                          j] = [j * self.circleDistance, i * self.circleDistance, 0]

        # move to center
        circleGridObjectPositions = circleGridObjectPositions - np.mean(circleGridObjectPositions, axis=0)
        return circleGridObjectPositions

    def calibOperation(self, allObjectPositions, raw_image_points, gray):
        flags = cv.CALIB_FIX_K3 | cv.CALIB_FIX_K2
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
        result = cv.calibrateCamera(allObjectPositions,
                                    raw_image_points,
                                    gray.shape[::-1],
                                    None,
                                    None,
                                    flags=flags,
                                    criteria=criteria)
        # ret, camera_matrix, dist_coeffs, rvecs, tvecs
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = result
        print(f'Camera matrix:\n{camera_matrix}')
        print(f'Distortion coefficients:\n{dist_coeffs}')
        return result

    def _refine_image_points_single(self, image_file, camera_matrix, dist_coeffs, rvec, tvec, obj_pts):
        image = cv.imread(image_file)
        undist = cv.undistort(image, camera_matrix, dist_coeffs)

        allObjX = obj_pts[:, 0]
        allObjY = obj_pts[:, 1]
        d = (np.max(allObjX) - np.min(allObjX)) / 6
        min_x, max_x = np.min(allObjX) - d, np.max(allObjX) + d
        min_y, max_y = np.min(allObjY) - d, np.max(allObjY) + d

        corners_obj = np.array([[min_x, min_y, 0], [min_x, max_y, 0], [max_x, max_y, 0], [max_x, min_y, 0]],
                               dtype=np.float32)

        proj_corners, _ = cv.projectPoints(corners_obj, rvec, tvec, camera_matrix, np.zeros_like(dist_coeffs))
        proj_corners = proj_corners.reshape(-1, 2).astype(np.float32)

        squareSize = 800
        dest_pts = np.array([[0, 0], [0, squareSize], [squareSize, squareSize], [squareSize, 0]], dtype=np.float32)
        H = cv.getPerspectiveTransform(proj_corners, dest_pts)
        H_inv = np.linalg.inv(H)

        canonical = cv.warpPerspective(undist, H, (squareSize, squareSize))

        # 检测圆心（在规范化的 canonical 图像中）
        centers = self._detectCircleGrid(canonical)

        # centers 形状: (N,1,2) — 处于 canonical 图像坐标系
        pts = centers.astype(np.float32).reshape(-1, 1, 2)

        # 反向透视变换到去畸变的原始图像坐标（undist 空间）
        img_pts = cv.perspectiveTransform(pts, H_inv)  # 返回 (N,1,2)
        img_pts = img_pts.reshape(-1, 2)

        # 添加畸变
        img_pts = distort_points_manual(img_pts, camera_matrix, dist_coeffs)
        return np.array(img_pts, dtype=np.float32)

    def calib(self):
        # 拆分为多个子函数以提高可读性
        def _initial_detect_and_calibrate(allImageFile):
            imageLength = len(allImageFile)
            objectPointsSingle = self.getObjectPoints()
            allObjectPositions = [objectPointsSingle for _ in range(imageLength)]

            raw_image_points = []
            for image_file in allImageFile:
                centers = self._detectCircleGrid(image_file)
                raw_image_points.append(centers)

            image0 = cv.imread(allImageFile[0])
            gray = cv.cvtColor(image0, cv.COLOR_BGR2GRAY)
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = self.calibOperation(allObjectPositions, raw_image_points,
                                                                                gray)

            return camera_matrix, dist_coeffs, rvecs, tvecs, allObjectPositions, gray

        def _iterative_refinement(allImageFile,
                                  camera_matrix,
                                  dist_coeffs,
                                  rvecs,
                                  tvecs,
                                  allObjectPositions,
                                  gray,
                                  max_iters=20):
            prev_err = None
            obj_pts = allObjectPositions[0]
            for it in range(max_iters):
                refined_image_points = []
                for idx, image_file in enumerate(allImageFile):
                    rvec = rvecs[idx].flatten() if isinstance(rvecs[idx], np.ndarray) else rvecs[idx]
                    tvec = tvecs[idx].flatten() if isinstance(tvecs[idx], np.ndarray) else tvecs[idx]
                    img_pts = self._refine_image_points_single(image_file, camera_matrix, dist_coeffs, rvec, tvec,
                                                               obj_pts)
                    refined_image_points.append(img_pts)

                ret, camera_matrix_new, dist_coeffs_new, rvecs_new, tvecs_new = self.calibOperation(
                    allObjectPositions, refined_image_points, gray)

                mean_err = ret
                print(f'Iteration {it}: calibration error = {mean_err:.6f}')
                if prev_err is not None and abs(prev_err - mean_err) < 1e-6:
                    camera_matrix, dist_coeffs, rvecs, tvecs = camera_matrix_new, dist_coeffs_new, rvecs_new, tvecs_new
                    break

                camera_matrix, dist_coeffs, rvecs, tvecs = camera_matrix_new, dist_coeffs_new, rvecs_new, tvecs_new
                prev_err = mean_err

            return camera_matrix, dist_coeffs, rvecs, tvecs

        # 主流程
        allImageFile = getAllImage(self.imageAndPoseDirectory)
        allImageFile = allImageFile[:self.maxImageLength]
        imageLength = len(allImageFile)

        camera_matrix, dist_coeffs, rvecs, tvecs, allObjectPositions, gray = _initial_detect_and_calibrate(allImageFile)
        camera_matrix, dist_coeffs, rvecs, tvecs = _iterative_refinement(allImageFile, camera_matrix, dist_coeffs,
                                                                         rvecs, tvecs, allObjectPositions, gray)

        np.savetxt('camera_matrix.txt', camera_matrix, fmt='%.6f', delimiter=',')
        np.savetxt('dist_coeffs.txt', dist_coeffs, fmt='%.6f', delimiter=',')

        all_rvecs = np.array(rvecs)
        gTc = self.calculateCalib(imageLength, allImageFile, all_rvecs, tvecs)
        np.savetxt('gTc.txt', gTc, fmt='%.6f', delimiter=',')
        return gTc

    def _detectCircleGridRefine(self, image_file):
        # 先使用 _detectCircleGrid 获取初始点位置
        centers = self._detectCircleGrid(image_file)
        img_pts = centers

        objPos = self.getObjectPoints()
        camera_matrix = self.camera_matrix
        dist_coeffs = self.dist_coeffs

        # 计算 rvec, tvec
        retval, rvec, tvec = cv.solvePnP(objPos, img_pts, camera_matrix, dist_coeffs)
        if not retval:
            raise Exception("solvePnP failed to compute rvec and tvec")

        # 调用细化函数
        return self._refine_image_points_single(image_file, camera_matrix, dist_coeffs, rvec, tvec, objPos)

    def loc(self, returnAllBTt=False):
        allImageFile = getAllImage(self.imageAndPoseDirectory)
        allImageFile = allImageFile[:self.maxImageLength]

        all_cTt = []
        imagePoints = []

        # 从每一张图片中获取 cTt
        for image_file in allImageFile:
            try:
                centers = self._detectCircleGridRefine(image_file)
                imagePoints.append(centers)
            except Exception as e:
                print(f"Error occurred while processing {image_file}: {e}")
                raise RuntimeError()

        objectPoints = self.getObjectPoints()

        camera_matrix = self.camera_matrix
        dist_coeffs = self.dist_coeffs

        # 计算 target 坐标系下的相机位姿
        for centers in imagePoints:
            cTt = self.calculate_cTt(objectPoints, centers, camera_matrix, dist_coeffs)
            all_cTt.append(cTt)

        all_bTt = self.calculateAllBTt(all_cTt, allImageFile)
        if returnAllBTt:
            return all_bTt

        imageLength = len(allImageFile)
        return self.analyzeBTt(all_bTt, imageLength, allImageFile)

    def locMinReproject(self, returnBTt=False, selectedViews=range(25)):
        allImageFile = getAllImage(self.imageAndPoseDirectory)
        allImageFile = allImageFile[:self.maxImageLength]

        imagePoints = []
        for index, image_file in enumerate(allImageFile):
            if index not in selectedViews:
                continue
            try:
                centers = self._detectCircleGridRefine(image_file)
                imagePoints.append(np.asarray(centers, dtype=np.float64))
            except Exception as e:
                print(f"Error occurred while processing {image_file}: {e}")
                raise RuntimeError()

        objectPoints = np.asarray(self.getObjectPoints(), dtype=np.float64)
        objectPointsCv = objectPoints.reshape(-1, 1, 3)

        camera_matrix = np.asarray(self.camera_matrix, dtype=np.float64)
        dist_coeffs = np.asarray(self.dist_coeffs, dtype=np.float64)

        all_bTg = read_all_bTg(self.imageAndPoseDirectory, allImageFile)
        gTc = self.gTc
        bTc_list = [bTg @ gTc for bTg in all_bTg]
        bTc_list = np.array(bTc_list)[selectedViews].copy()
        imagePoints = np.array(imagePoints)
        print(f'bTc shape: {bTc_list.shape}, image points shape: {imagePoints.shape}')

        all_bTt_init = []
        for centers, bTc in zip(imagePoints, bTc_list):
            cTt = self.calculate_cTt(objectPoints, centers, camera_matrix, dist_coeffs)
            all_bTt_init.append(bTc @ cTt)

        init_bTt = self.meanMatrix(all_bTt_init)
        init_rvec, _ = cv.Rodrigues(init_bTt[:3, :3])
        x0 = np.zeros(6, dtype=np.float64)
        x0[:3] = init_rvec.flatten()
        x0[3:] = init_bTt[:3, 3]

        q_obs = np.asarray(imagePoints, dtype=np.float64)

        def residuals(x):
            bTt = rtToMatrix(x[:3], x[3:6])
            res = []
            for i, bTc in enumerate(bTc_list):
                cTt = np.linalg.inv(bTc) @ bTt
                rvec_ct, _ = cv.Rodrigues(cTt[:3, :3])
                tvec_ct = cTt[:3, 3].reshape(3, 1)

                imgpts, _ = cv.projectPoints(objectPointsCv, rvec_ct, tvec_ct, camera_matrix, dist_coeffs)
                imgpts = imgpts.reshape(-1, 2)

                obs = q_obs[i]
                valid = ~np.isnan(obs).any(axis=1)
                diff = imgpts[valid] - obs[valid]
                res.extend(diff.reshape(-1))
            return np.asarray(res, dtype=np.float64)

        res0 = residuals(x0)
        print(f'Residual shape: {res0.shape}')
        method = 'lm' if res0.size >= x0.size else 'trf'
        optimize_result = least_squares(residuals, x0, method=method)

        opt_x = optimize_result.x
        bTt_opt = rtToMatrix(opt_x[:3], opt_x[3:6])

        print('locMinReproject optimize status:', optimize_result.status, optimize_result.message)
        print('locMinReproject residual norm:', np.linalg.norm(optimize_result.fun))

        # 兼容现有 analyzeBTt 接口：为每帧恢复对应的 bTt（理论上应接近同一个解）
        all_bTt = []
        for bTc in bTc_list:
            cTt_opt = np.linalg.inv(bTc) @ bTt_opt
            all_bTt.append(bTc @ cTt_opt)
        all_bTt = np.array(all_bTt)
        bTt = all_bTt[0]

        if returnBTt:
            return bTt

        imageLength = len(allImageFile)
        return self.analyzeBTt(all_bTt, imageLength, allImageFile)

    def oneloc(self):
        image_path = self.imageAndPoseDirectory + self.imageRelativeFile

        # 所有 circleGrid 的点位置
        centers = self._detectCircleGrid(image_path)

        # 获得 target 坐标系下的 circleGrid 的点位置
        object_points = self.getObjectPoints()
        newObjectPoints, rectanglePoints, center = movePointsToRectangleCenter(object_points)
        image_points = centers

        # 计算 target 坐标系下的相机位姿
        result = self.calculateOneloc(newObjectPoints, image_points)
        print('base to target 的齐次矩阵是：')
        print(result)
        return result


if __name__ == '__main__':
    # 读取参数 config.json
    assert len(sys.argv) == 2
    file = sys.argv[1]
    config = readConfig(file)

    type = config['type']
    task = config['task']

    if type == 'charuco':
        pattern = Charuco(config)
    elif type == 'aruco':
        pattern = Aruco(config)
    elif type == 'circleGrid':
        pattern = CircleGrid(config)

    if task == 'oneloc':
        result = pattern.oneloc()
    elif task == 'calib':
        result = pattern.calib()
    elif task == 'loc':
        result = pattern.loc()
    elif task == 'locMinReproject':
        result = pattern.locMinReproject()
