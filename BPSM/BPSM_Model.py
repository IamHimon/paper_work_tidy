import numpy as np
import pandas as pd
import json

from tools import LABEL_DICT


class BPSM():
    def __init__(self, dict_of_domain: dict):
        self.Domain = dict_of_domain
        self.Attr_len = len(self.Domain)
        self.FTn = np.zeros((self.Attr_len, self.Attr_len))  # 正向序列统计次数, FTn[i][j] 表示Li->Lj
        self.FTp = pd.DataFrame(np.zeros((self.Attr_len, self.Attr_len)))  # 正向序列统计概率
        self.BTn = np.zeros((self.Attr_len, self.Attr_len))  # 反向序列统计次数, BTn[j][i] 表示Lj->Li
        self.BTp = pd.DataFrame(np.zeros((self.Attr_len, self.Attr_len)))  # 反向序列统计概率
        self.PSn = np.zeros((self.Attr_len, self.Attr_len))  # 位置统计次数,  PSn[i][j], Li出现在位置j
        self.PSp = pd.DataFrame(np.zeros((self.Attr_len, self.Attr_len)))  # 位置统计概率

    # 　逐条学习BPSM模型
    def step_learnBPSM(self, prediction: list):
        for i in range(len(prediction)):
            if i < len(prediction) - 1:
                # 正向统计序列转化次数,从prediction[i]-> prediction[i+1]
                self.FTn[prediction[i]][prediction[i + 1]] += 1.0
                # 反向统计序列转化次数 从prediction[i + 1] -> prediction[i]
                self.BTn[prediction[i + 1]][prediction[i]] += 1.0
            # 统计标签在每个位置出现的次数,在i位置出现prediction[i]
            self.PSn[prediction[i]][i] += 1.0

    # 将TFn, BTn, PSp 转化为 TFp, BTp, PSp, 也就是次数转化为概率
    def convertM(self):
        FTn = pd.DataFrame(self.FTn)
        BTn = pd.DataFrame(self.BTn)
        PSn = pd.DataFrame(self.PSn)
        self.FTp = FTn.div(FTn.sum(axis=1), axis=0)
        self.BTp = BTn.div(BTn.sum(axis=1), axis=0)
        self.PSp = PSn.div(PSn.sum(axis=1), axis=0)

    # 从实验结果(json格式存储)中学习BPSM
    def learnBPSM_from_json(self, filename):
        f = open(filename, "r")
        for line in f:
            decodes = json.loads(line)
            # print(decodes['ID'])
            # print(decodes['predictions'])
            # 如果有'None'就去掉
            predictions = [int(p) for p in decodes['predictions'] if (p is not None and p != 'None')]
            # print(predictions)
            self.step_learnBPSM(predictions)
            # print('=========')


if __name__ == '__main__':
    bpsm = BPSM(LABEL_DICT)
    bpsm.learnBPSM_from_json('result_0.96.json')
    print(bpsm.BTn)
    bpsm.convertM()
    print(bpsm.BTp)

    # bpsm = BPSM(LABEL_DICT)
    # bpsm.learn_BPSM([0, 1, 2, 3, 4, 5])
    # bpsm.learn_BPSM([5, 2, 1, 3, 4, 0])
    # bpsm.learn_BPSM([0, 2, 1, 3, 4, 5])
    # print(bpsm.FTp)
    # bpsm.convertM()
    # print(bpsm.FTp)
