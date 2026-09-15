import matrix_utils

import numpy as np
import os
from natsort import natsorted
import pinocchio as pin
import json
import pandas as pd
import matplotlib

matplotlib.use('Agg')

from matplotlib import pyplot as plt
import matplotlib.colors as colors
import seaborn as sns

# 设置 pd 两位小数
pd.set_option('display.precision', 2)


def averageAllMatrix(allMatrixT):
    allT = []
    allQ = []
    for T in allMatrixT:
        tq = matrix_utils.matrixToTQ(T)
        allT.append(tq[:3])
        allQ.append(tq[3:])
    avgT = np.mean(np.array(allT), axis=0)
    A = sum([np.outer(q, q) for q in allQ])
    # 计算最大特征值对应的特征向量
    eigenvalues, eigenvectors = np.linalg.eig(A)
    max_index = np.argmax(eigenvalues)
    avgQ = eigenvectors[:, max_index]

    if not np.isreal(avgQ).all():
        print("Warning: avgQ is not real:", avgQ)
    # 去除虚数部分
    avgQ = np.real(avgQ)
    # 如果不是实数，或者模长不为 1，则发出警告
    if not np.isclose(np.linalg.norm(avgQ), 1.0):
        print("Warning: avgQ norm is not 1:", np.linalg.norm(avgQ))

    Tref = matrix_utils.tqToMatrix(np.concatenate([avgT, avgQ]))
    return Tref


def traditionalMultiView(trainData):
    N = len(trainData)
    M = trainData[0].shape[0]
    assert trainData.shape == (N, M, 6)
    # 计算 N M 个位姿的平移融合，得到一个 t_ref
    allMatrixT = []
    for vec in trainData.reshape(-1, 6):
        T = matrix_utils.trToMatrix(vec)
        allMatrixT.append(T)

    Tref = averageAllMatrix(allMatrixT)
    return Tref


class TraditionalMultiView:
    def __init__(self, debug=True):
        self.debug = debug

    def train(self, trainData):
        N = len(trainData)
        M = trainData[0].shape[0]
        assert trainData.shape == (N, M, 6)
        if self.debug:
            print('N:', N, 'M:', M)

        Tref = traditionalMultiView(trainData)
        self.Tref = Tref
        self.M = M
        return Tref

    def infer(self, testData):
        N_test = len(testData)
        M_test = testData[0].shape[0]
        assert M_test == self.M
        assert N_test == 1
        assert testData.shape == (N_test, M_test, 6)
        if self.debug:
            print('N_test:', N_test, 'M_test:', M_test)

        Tref_test = traditionalMultiView(testData)
        return Tref_test


def getPoseFromPosition(position):
    mTt = np.eye(4)
    mTt[0:3, 3] = position
    return mTt


def _rotation_angle_urad(R):
    """Return rotation angle (axis-angle magnitude) in urad."""
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta_rad = float(np.arccos(cos_theta))
    return theta_rad * 1e6


def multiViewAccuracy(multiViewMethod, trainData, testData, position, debug=True):
    N = len(trainData)
    M = trainData[0].shape[0]
    assert trainData.shape == (N, M, 6)
    if debug:
        print('N:', N, 'M:', M)

    position = np.array(position)
    assert position.shape == (3, ) or position.shape[1] == 3
    if position.shape == (3, ):
        position = position.reshape(1, 3)

    multiViewObject = multiViewMethod(debug=debug)
    trained_T = multiViewObject.train(trainData)

    # 对于每一个 testData 计算 eachTest_T_traditional
    all_diff_xyz = []
    all_diff_angle_urad = []
    for i in range(len(testData)):
        eachTest_T = multiViewObject.infer(np.array([testData[i]]))
        for j in range(len(position)):
            pos = position[j]
            mTt = getPoseFromPosition(pos)
            bTt_train = trained_T @ mTt
            bTt_test = eachTest_T @ mTt
            diff = np.linalg.inv(bTt_train) @ bTt_test
            all_diff_xyz.append(diff[0:3, 3] * 1000)
            all_diff_angle_urad.append(_rotation_angle_urad(diff[:3, :3]))

    all_diff_xyz = np.array(all_diff_xyz)
    all_diff_angle_urad = np.array(all_diff_angle_urad)
    if debug:
        mean_diff = np.mean(all_diff_xyz, axis=0)
        std_diff = np.std(all_diff_xyz, axis=0)
        print("Mean difference (um):", mean_diff)
        print("Std difference (um):", std_diff)

        mean_angle = np.mean(all_diff_angle_urad)
        std_angle = np.std(all_diff_angle_urad)
        print("Mean angle (urad):", mean_angle)
        print("Std angle (urad):", std_angle)

    rmseX = np.sqrt(np.mean(all_diff_xyz[:, 0]**2))
    rmseY = np.sqrt(np.mean(all_diff_xyz[:, 1]**2))
    rmseZ = np.sqrt(np.mean(all_diff_xyz[:, 2]**2))
    rmseAngle = np.sqrt(np.mean(all_diff_angle_urad**2))
    if debug:
        print("RMSE (um):", [rmseX, rmseY, rmseZ])
        print("RMSE angle (urad):", rmseAngle)
    return rmseX, rmseY, rmseZ, rmseAngle


# 使用加权融合方法计算多视角融合精度
class WeightedMultiView:
    def __init__(self, debug=True):
        self.debug = debug

    def _computeTrefAndTwist(self, data, N, M):
        assert data.shape == (N, M, 6)
        # 首先全部转为位姿矩阵，计算 Tref
        allMatrixT = []
        for vec in data.reshape(-1, 6):
            T = matrix_utils.trToMatrix(vec)
            allMatrixT.append(T)
        Tref = averageAllMatrix(allMatrixT)
        assert np.array(allMatrixT).shape == (N * M, 4, 4)

        # 计算每个视角的误差矩阵 E
        allE = []
        for T in allMatrixT:
            E = np.linalg.inv(Tref) @ T
            allE.append(E)
        allE = np.array(allE)
        allE = allE.reshape(N, M, 4, 4)
        allTwist = np.array([[pin.log6(pin.SE3(allE[i, j])).vector for j in range(M)] for i in range(N)])
        return Tref, allTwist

    def _fuse(self, allAvgTwist, allInformationMatrix, M):
        # 计算融合结果
        denominator = np.sum(allInformationMatrix, axis=0)
        numerator = np.sum([allInformationMatrix[i] @ allAvgTwist[i] for i in range(M)], axis=0)
        avgTwist = np.linalg.inv(denominator) @ numerator
        return avgTwist

    def train(self, trainData):
        N = len(trainData)
        M = trainData[0].shape[0]
        assert trainData.shape == (N, M, 6)
        if self.debug:
            print('N:', N, 'M:', M)

        Tref, allTwist = self._computeTrefAndTwist(trainData, N, M)

        # 按照视角计算平均 twist
        allAvgTwist = np.mean(allTwist, axis=0)
        assert allAvgTwist.shape == (M, 6)

        # 计算每个视角的协方差矩阵，dim = (M, 6, 6)
        allInformationMatrix = []
        for view_idx in range(M):
            cov_matrix = np.cov(allTwist[:, view_idx].T)
            info_matrix = np.linalg.inv(cov_matrix)

            if self.debug and selectedViews[view_idx] in [0, 12]:
                print(f"视角 {selectedViews[view_idx]} 的协方差矩阵:\n{cov_matrix}")
            allInformationMatrix.append(info_matrix)
        allInformationMatrix = np.array(allInformationMatrix)
        assert allInformationMatrix.shape == (M, 6, 6)

        # 强制设置为对角矩阵，看是否退化
        # allInformationMatrix = np.array([np.diag(np.diag(mat)) for mat in allInformationMatrix])

        allInformationMatrix = np.array(allInformationMatrix)
        assert allInformationMatrix.shape == (M, 6, 6)

        # 计算融合结果
        avgTwist = self._fuse(allAvgTwist, allInformationMatrix, M)
        diff_fused = pin.Motion(avgTwist)
        delta_T = pin.exp6(diff_fused)
        Tfused = Tref @ delta_T

        # 存储必要的信息 N M InformationMatrix Tfused
        self.M = M
        self.allInformationMatrix = allInformationMatrix
        self.Tfused = Tfused
        return Tfused

    def infer(self, testData):
        N_test = len(testData)
        M_test = testData[0].shape[0]
        assert M_test == self.M
        assert N_test == 1
        assert testData.shape == (N_test, M_test, 6)
        if self.debug:
            print('N_test:', N_test, 'M_test:', M_test)

        # 先计算 Tref_test and allTwist_test
        Tref_test, allTwist_test = self._computeTrefAndTwist(testData, N_test, M_test)

        # 计算每个视角的平均 twist
        allAvgTwist_test = np.mean(allTwist_test, axis=0)
        assert allAvgTwist_test.shape == (M_test, 6)

        # 计算融合结果
        avgTwist_test = self._fuse(allAvgTwist_test, self.allInformationMatrix, M_test)
        diff_fused_test = pin.Motion(avgTwist_test)
        delta_T_test = pin.exp6(diff_fused_test)
        Tfused_test = Tref_test @ delta_T_test
        return Tfused_test


class MinReprojectMultiView:
    def __init__(self, debug=False):
        self.debug = debug

    def train(self, trainData):
        N = len(trainData)
        assert trainData.shape == (N, 1, 6)
        # 转成 mat
        allMatrixT = []
        for i in range(N):
            matrixT = matrix_utils.trToMatrix(trainData[i][0])
            allMatrixT.append(matrixT)

        # 使用 averageAllMatrix 计算平均 pose
        avg = averageAllMatrix(allMatrixT)
        self.avg = avg
        return avg

    def infer(self, testData):
        N = len(testData)
        assert N == 1
        assert testData.shape == (1, 1, 6)
        mat = matrix_utils.trToMatrix(testData[0][0])
        return mat


class FullLocMinReprojectMultiView:
    """使用 locMinReproject-60.py 的联合优化结果作为 train，单组 minReproject 作为 test。"""
    def __init__(self, debug=False):
        self.debug = debug
        num_views = len(selectedViews)
        json_path = f'result-json/minReproject-60-{num_views}.json'
        full_result = json.load(open(json_path))

        # --- 校验：视角数量和具体视角一致 ---
        json_views = full_result['selectedViews']
        assert json_views == selectedViews

        # --- 校验：文件夹列表一致 ---
        json_folders = full_result['folders']
        expected_folders = allFolders[:full_result['numFolders']]
        assert json_folders == expected_folders

        self.full_bTt = matrix_utils.trToMatrix(np.array(full_result['bTt_tr']))
        if debug:
            print(f'FullLocMinReprojectMultiView 加载 {json_path}，校验通过')
            print(f'  文件夹数: {full_result["numFolders"]}, 视角: {json_views}')
            print(f'full_bTt:\n{self.full_bTt}')

    def train(self, trainData):
        """忽略 trainData，直接返回全局联合优化的 bTt。"""
        return self.full_bTt

    def infer(self, testData):
        """每个 test 样本是单组的 minReproject 结果。"""
        N = len(testData)
        assert N == 1
        assert testData.shape == (1, 1, 6)
        mat = matrix_utils.trToMatrix(testData[0][0])
        return mat


# 测试视角数量与精度的关系
def testViewNumberAccuracy(multiViewMethod,
                           viewNumbers,
                           repeatTimes=5,
                           position=[400, 0, 0],
                           debug=True,
                           viewRange=range(25),
                           trainStartIndex=0,
                           trainLength=60,
                           testLength=60):
    # 设置 seed
    np.random.seed(0)
    # 随机选择 0 到 24 号中的 viewNumbers 个视角，进行多视角融合，计算精度
    # 重复 repeatTimes，将每次的 rmseX, rmseY, rmseZ 组合成新的 rmse
    results = {}
    for numViews in viewNumbers:
        rmseXList = []
        rmseYList = []
        rmseZList = []
        rmseAngleList = []

        tmpRepeatTimes = repeatTimes
        if numViews == len(viewRange):
            tmpRepeatTimes = 1

        for _ in range(tmpRepeatTimes):
            selectedIndices = np.random.choice(len(viewRange), numViews, replace=False)
            selectedViewIndices = [viewRange[i] for i in selectedIndices]
            selectedMultiViewData = []
            for folder in allFolders:
                if multiViewMethod in (MinReprojectMultiView, FullLocMinReprojectMultiView):
                    selectedData = all_results_minReproject[folder]
                else:
                    selectedData = all_results[folder][selectedViewIndices]

                selectedMultiViewData.append(selectedData)
            selectedMultiViewData = np.array(selectedMultiViewData)

            assert trainStartIndex + trainLength + testLength <= len(selectedMultiViewData)
            trainData = selectedMultiViewData[trainStartIndex:trainStartIndex + trainLength]
            testData = selectedMultiViewData[-testLength:]
            if debug:
                print("numViews:", numViews, "selectedViewIndices:", selectedViewIndices)
            rmseX, rmseY, rmseZ, rmseAngle = multiViewAccuracy(multiViewMethod,
                                                               trainData,
                                                               testData,
                                                               position=position,
                                                               debug=debug)

            rmseXList.append(rmseX)
            rmseYList.append(rmseY)
            rmseZList.append(rmseZ)
            rmseAngleList.append(rmseAngle)
        # 使用均方根作为最终结果
        finalRmseX = np.sqrt(np.mean(np.array(rmseXList)**2))
        finalRmseY = np.sqrt(np.mean(np.array(rmseYList)**2))
        finalRmseZ = np.sqrt(np.mean(np.array(rmseZList)**2))
        finalRmseAngle = np.sqrt(np.mean(np.array(rmseAngleList)**2))
        finalRMSE = np.sqrt(finalRmseX**2 + finalRmseY**2 + finalRmseZ**2)
        results[numViews] = {
            'rmseX': np.sqrt(np.mean(np.array(rmseXList)**2)),
            'rmseY': np.sqrt(np.mean(np.array(rmseYList)**2)),
            'rmseZ': np.sqrt(np.mean(np.array(rmseZList)**2)),
            'RMSE': finalRMSE,
            'angle': finalRmseAngle,
        }
    finalSeed = np.random.randint(0, 100)
    # 结果用 pandas 打印
    resultDf = pd.DataFrame(results).T
    return resultDf, finalSeed


def compareTraditionalAndWeighted(viewNumbers, repeatTimes, position, viewRange=range(25)):
    enableMinReproject = False
    if len(viewNumbers) == 1:
        if len(selectedViews) == len(viewRange) == viewNumbers[0]:
            if list(selectedViews) == list(viewRange):
                enableMinReproject = True

    if enableMinReproject:
        minReprojectDf, minReprojectFinalSeed = testViewNumberAccuracy(MinReprojectMultiView,
                                                                       viewNumbers,
                                                                       repeatTimes,
                                                                       position,
                                                                       debug=False,
                                                                       viewRange=viewRange)
        fullMinReprojectDf, fullMinReprojectFinalSeed = testViewNumberAccuracy(FullLocMinReprojectMultiView,
                                                                               viewNumbers,
                                                                               repeatTimes,
                                                                               position,
                                                                               debug=False,
                                                                               viewRange=viewRange)
    # 传统多视角融合结果
    resultDf, traditionalFinalSeed = testViewNumberAccuracy(TraditionalMultiView,
                                                            viewNumbers,
                                                            repeatTimes,
                                                            position,
                                                            debug=False,
                                                            viewRange=viewRange)
    # 加权多视角融合结果
    weightedResultDf, weightedFinalSeed = testViewNumberAccuracy(WeightedMultiView,
                                                                 viewNumbers,
                                                                 repeatTimes,
                                                                 position,
                                                                 debug=False,
                                                                 viewRange=viewRange)
    if enableMinReproject:
        assert traditionalFinalSeed == weightedFinalSeed == minReprojectFinalSeed == fullMinReprojectFinalSeed
        # 拼接 resultDf 和 weightedResultDf 的所有列
        combinedDf = pd.concat([
            resultDf.add_prefix('T_'),
            weightedResultDf.add_prefix('W_'),
            minReprojectDf.add_prefix('R_'),
            fullMinReprojectDf.add_prefix('F_')
        ],
                               axis=1)
    else:
        assert traditionalFinalSeed == weightedFinalSeed
        # 拼接 resultDf 和 weightedResultDf 的所有列
        combinedDf = pd.concat([resultDf.add_prefix('T_'), weightedResultDf.add_prefix('W_')], axis=1)
    # 计算 RMSE 比例
    return combinedDf


def testEach():
    position = [[-400, 0, 0], [400, 0, 0], [0, -400, 0], [0, 400, 0]]

    repeatTimes = 100
    # 对于每个视角，计算每个 T_RMSE 与 W_RMSE，给出表格
    eachViewData = []
    for viewIdx in range(25):
        tmpDf = compareTraditionalAndWeighted([1], repeatTimes, position, viewRange=[viewIdx])
        eachViewData.append(tmpDf)
    eachViewDf = pd.concat(eachViewData)
    # reset index
    eachViewDf.reset_index(inplace=True)

    # plot RMSE
    allRMSE400 = eachViewDf['T_RMSE'].to_numpy(copy=True).reshape(5, 5)
    allRMSE400[1] = allRMSE400[1][::-1]
    allRMSE400[3] = allRMSE400[3][::-1]

    print()
    print('single-view localization RMSE (um), arranged in a 5 x 5 view grid')
    print(np.round(allRMSE400, 1))
    plot_elegant_heatmap(allRMSE400)
    return eachViewDf


def plot_elegant_heatmap(data):
    plt.figure(figsize=(8, 7))

    # 将数据转为 DataFrame 方便标注坐标
    df = pd.DataFrame(data, index=[-10, -5, 0, 5, 10], columns=[-10, -5, 0, 5, 10])

    # 使用更加柔和的颜色映射 (如 'YlGnBu' 或 'rocket_r')
    # annot=True 会在格子上写数字，fmt=".1f" 保留一位小数
    sns.heatmap(df,
                annot=True,
                fmt=".1f",
                cmap='magma_r',
                linewidths=.5,
                cbar_kws={'label': 'RMSE (μm)'},
                norm=colors.LogNorm())

    plt.title('Single-View Localization Error Heatmap', fontsize=14, pad=20)
    plt.xlabel('X Offset (cm)')
    plt.ylabel('Y Offset (cm)')
    plt.savefig('rmse_annotated_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()


def runEvaluate(allDistance):
    global selectedViews, all_results_minReproject
    viewNumberAndRange = [
        ([1], [12]),
        ([1], [0]),
        ([4], [0, 4, 20, 24]),
        ([9], [0, 2, 4, 10, 12, 14, 20, 22, 24]),
        # ([25], range(25)),
    ]
    allCombinedDf = pd.DataFrame()
    for distance in allDistance:
        position = [[-distance, 0, 0], [distance, 0, 0], [0, -distance, 0], [0, distance, 0]]
        repeatTimes = 100

        for viewNumbers, viewRange in viewNumberAndRange:
            num_views = viewNumbers[0]
            # 多视角时按视角数加载对应的 minReproject JSON（供 R_ / F_ 方法使用）
            if num_views > 1:
                minReprojectFile = f'{rootFolder}/minReproject-{num_views}.json'
                minReproject_data = json.load(open(minReprojectFile))
                selectedViews = minReproject_data['selectedViews']
                all_results_minReproject = minReproject_data['data']
                for folder in allFolders:
                    all_results_minReproject[folder] = np.array(all_results_minReproject[folder])

            combinedDf = compareTraditionalAndWeighted(viewNumbers, repeatTimes, position, viewRange).copy()

            # 对于单视角，把 W_ 开头的都设置为 NaN
            if viewNumbers == [1]:
                cols_to_fix = combinedDf.columns[combinedDf.columns.str.startswith('W_')]
                combinedDf[cols_to_fix] = np.nan

            # 对于非 0 的 distance，把 angle 结尾的列设置为 NaN
            if distance != 0:
                cols_to_fix = combinedDf.columns[combinedDf.columns.str.endswith('angle')]
                combinedDf[cols_to_fix] = np.nan

            # distance 列排在最前面
            combinedDf['distance'] = distance
            combinedDf = combinedDf[['distance'] + [col for col in combinedDf.columns if col != 'distance']]

            allCombinedDf = pd.concat([allCombinedDf, combinedDf])
    print(allCombinedDf)


def main():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    runEvaluate([0, 400])
    eachViewDf = testEach()
    print()
    print(eachViewDf)


dataFolder = 'data/calib-150-times'
allFolders = natsorted(os.listdir(dataFolder))
allFolders = [f'{dataFolder}/{f}/' for f in allFolders]

rootFolder = 'result-json'
allResultsFile = rootFolder + '/allBTt.json'
allResultsMinReprojectFile = rootFolder + '/minReproject-4.json'
# allResultsMinReprojectFile = rootFolder + '/minReproject-9.json'

all_results = json.load(open(allResultsFile))
for folder in allFolders:
    all_results[folder] = np.array(all_results[folder])

minReproject = json.load(open(allResultsMinReprojectFile))
selectedViews = minReproject['selectedViews']
all_results_minReproject = minReproject['data']
for folder in allFolders:
    all_results_minReproject[folder] = np.array(all_results_minReproject[folder])

main()
