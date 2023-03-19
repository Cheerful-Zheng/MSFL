import numpy as np
from pyod.models.iforest import IForest
from sklearn.cluster import DBSCAN


class State:
    """
    用于存储每个客户状态，包括训练权重、恶意因子与分组
    """
    # TODO 调整参数
    __xi = 1/1000
    __l = 1
    __V = 0.5
    h = 0.01

    def __init__(self):
        self.name = None
        self.currentWeight = []
        self.client_state = None
        self.data = None

        self.error = []
        self.accumulatedFactor = 0

        self.flag = []
        self.continuousFactor = 0

        self.creditScore = 0
        self.group = 0

        self.p = 1

    def __init__(self, name, weight, client_state, data):
        self.name = name
        self.currentWeight = weight
        self.client_state = client_state
        self.data = data

        self.error = []
        self.accumulatedFactor = 0

        self.flag = []
        self.continuousFactor = 0

        self.creditScore = 0
        self.group = 0

        self.p = 1

    # def __init__(self, name, currentWeight: list, error: list, accumulatedFactor, flag: list, continuousFactor, group):
    #     self.name = name
    #     self.currentWeight = currentWeight
    #
    #     self.error = error
    #     self.accumulatedFactor = accumulatedFactor
    #
    #     self.flag = flag
    #     self.continuousFactor = continuousFactor
    #
    #     self.creditScore = self.getCreditScore()
    #     self.group = group



    def get_h(self):
        return self.h

    def get_data(self):
        return self.data

    def set_client_state(self, client_state):
        self.client_state = client_state

    def get_client_state(self):
        return self.client_state

    def calculateCreditScore(self):
        """
        计算恶意指数
        :return:恶意指数
        """
        self.creditScore = np.exp((1 - State.__V) * self.accumulatedFactor + State.__V * self.continuousFactor)
        return self.creditScore

    def get_credict_score(self):
        return self.creditScore

    def getAccumulatedFactor(self):
        return self.accumulatedFactor

    def setAccumulatedFactor(self):
        """
        计算累计恶意指数
        :return:累计恶意指数
        """
        sumErr = sum(self.error)
        factor = State.__xi * sumErr
        if factor > 0:
            self.accumulatedFactor = factor / len(self.error)
        else:
            self.accumulatedFactor = 0

    def getContinuousFactor(self):
        return self.continuousFactor

    def setContinuousFactor(self):
        """
        计算持续恶意指数
        :return:持续恶意指数
        """
        sumFlag = sum(self.flag)
        factor = len(self.flag) - State.__l * sumFlag
        if factor > 0:
            self.continuousFactor = factor / len(self.flag)
        else:
            self.continuousFactor = 0

    def setGroup(self, group):
        self.group = group

    def getGroup(self):
        return self.group

    def getWeight(self):
        return self.currentWeight

    def setWeight(self, weight: list):
        self.currentWeight = weight

    def addErr(self, currentErr):
        self.error.append(currentErr)
        self.setAccumulatedFactor()

    def addFlag(self, currentFlag):
        self.flag.append(currentFlag)
        self.setContinuousFactor()

    def set_p(self, sum, ni):
        self.p = ni * (self.creditScore ** (-1 * self.h)) / sum

    def alter_weights(self, server_weight):
        for component in range(len(self.currentWeight)):
            self.currentWeight[component] = self.p * (self.currentWeight[component] - server_weight[component]) + \
                                            server_weight[component]

        self.client_state.local_weights[0].assign(np.array([self.currentWeight]))



def getGroupError(groupStates, serverState: State):
    """
    计算当前分组用户的累积恶意因子，即该组用户的权重与服务器权重的平方根误差之和
    :param groupStates: 该分组用户的状态信息
    :param serverState: 服务器端的状态信息
    :return: 平方根误差
    """
    serverWeight = serverState.getWeight()
    err = 0
    for state in groupStates:
        if len(state.currentWeight) != len(serverWeight):
            print("服务器与用户权重长度不符")
            exit()

        # 计算该组用户中每个用户权值与服务器权值的平方根误差
        currentErr = 0
        for j in range(len(state.currentWeight)):
            currentErr = currentErr + (state.currentWeight[j] - serverWeight[j]) ** 2

        err = err + (currentErr / len(state.currentWeight)) ** 0.5

    return err


def trainOutlierClassifier(weight, serverState: State):
    """
    训练离群值检测器，即Flag求解器
    :param serverState:服务器的状态信息
    :return:离群值检测器
    """
    weight.append(serverState.getWeight())

    # 训练一个DBSCAN检测器
    clf = DBSCAN(eps=3.1, min_samples=2)
    clf.fit(weight)  # 使用X_train训练检测器clf

    return clf


def classifyOutlier(clf, groupStates):
    """
    离群值检测，即Flag函数
    :param clf: 离群值检测器
    :param groupStates: 该分组的状态信息
    :return: 该组数据是否正常
    """
    weight = []
    for client in groupStates:
        weight.append(client.getWeight())

    # 用训练好的clf来预测未知数据中的异常值，-1为离群值
    prediction = clf.fit_predict(weight)

    outlierFlag = -1
    print(prediction)
    if outlierFlag in prediction:
        return 0
    else:
        return 1
