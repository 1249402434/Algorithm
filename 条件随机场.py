import numpy as np

class CRF:
    def __init__(self,M,start,stop,y_status,y_start,y_stop):
        self.Z=0
        self.M=M
        #0态到终态
        self.length=stop-start
        #0态值和终态值
        self.y_start=y_start
        self.y_stop=y_stop
        #矩阵维度
        self.dim=len(y_status)

    def calc_Z(self):
        cumulative_mul = self.M[0]
        for i in range(1,self.length):
            cumulative_mul=cumulative_mul.dot(self.M[i])

        self.Z=cumulative_mul[0,0]
        return cumulative_mul[0,0]
    #计算特定路径的概率
    def calc_probaility(self,Y):
        num=len(Y)
        result=1
        x=0
        for i in range(num):
            y=Y[i]
            m=self.M[i]
            result*=m[x,y-1]
            x=y-1

        return result

M1 = np.array([[0.5, 0.5],[0,0]])
M2 = np.array([[0.3, 0.7],[0.7, 0.3]])
M3 = np.array([[0.5, 0.5],[0.6, 0.4]])
M4 = np.array([[1, 0],[1, 0]])
M=[]
M.append(M1)
M.append(M2)
M.append(M3)
M.append(M4)

M=np.array(M)

crf = CRF(M,0,4,[1,2],1,1)
Z=crf.calc_Z()
prob=crf.calc_probaility([1,2,1,1])
print('规范化因子：',Z)
print('非规范化概率：',prob)
print('规范化概率：',prob/Z)
