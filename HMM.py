import numpy as np

#例10.3
Q = [1, 2, 3]
V = ['红', '白']
A = [[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]
B = [[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]]
O = ['红', '白', '红']
PI = [0.2, 0.4, 0.4]


class HiddenMarkov:
    def __init__(self,Q,V,O):
        self.Q=Q
        self.V=V
        self.O=O
        self.N=len(Q)
        self.M=len(O)

    def forward(self,A,B,PI):
        print('------forward------')
        #前向概率alpha，每一时刻都存在一个长N的向量，故利用矩阵保存
        alphas=np.zeros((self.N,self.M))
        T=self.M
        for t in range(T):
            #找到此观测对应的索引
            index=self.V.index(self.O[t])
            for i in range(self.N):
                if t==0:
                    alphas[i][t]=PI[i]*B[i][index]
                    print('alpha1(%d)=pi%db%d(o1)=%f' % (i, i, i, alphas[i][t]))
                else:
                    alphas[i][t]=np.sum([alphas[j][t-1]*A[j][i] for j in range(self.N)])*B[i][index]
                    print('alpha%d(%d)=[sigma alpha%d(i)ai%d]b%d(o%d)=%f' % (t, i, t - 1, i, i, t, alphas[i][t]))

        P=np.sum([alpha[self.M-1] for alpha in alphas])
        print('probaility:',P)

    def backward(self,A,B,PI):
        print('------backward------')
        belta=np.zeros((self.N,self.M))
        T=self.M
        '''在T时刻往前推，时间是从0开始的'''
        for t in range(T-1,-1,-1):
            if t<T-1:
                index=self.V.index(self.O[t+1])
            for i in range(self.N):
                if t==T-1:
                    belta[i][t]=1
                    print('beltaT(%d)=%f'%(i,belta[i][t]))
                else:
                    belta[i][t]=np.sum([A[i][j]*B[j][index]*belta[j][t+1] for j in range(self.N)])
                    print('belta(%d)(%d)=%f'%(t,i,belta[i][t]))

        P=np.sum([PI[i]*B[i][self.V.index(O[0])]*belta[i][0] for i in range(self.N)])
        print('probaility:',P)

    #维特比算法
    def viterbi(self,A,B,PI):
        print('------viterbi------')
        delta=np.zeros((self.N,self.M))
        psis=np.zeros((self.N,self.M))
        T=self.M
        for t in range(T):
            index=self.V.index(O[t])
            for i in range(self.N):
                if t==0:
                    delta[i][t]=PI[i]*B[i][index]
                    psis[i][t]=0
                    print('delta1(%d)=%f'%(i,delta[i][t]))
                    print('psis1(%d)=%f'%(i,psis[i][t]))

                else:
                    max_value=np.max([delta[j][t-1]*A[j][i] for j in range(self.N)])
                    delta[i][t]=max_value*B[i][index]

                    psis[i][t]=np.argmax([delta[j][t-1]*A[j][i] for j in range(self.N)])

                    print('delta(%d)(%d)=%f'%(t,i,delta[i][t]))
                    print('psis(%d)(%d)=%f'%(t,i,psis[i][t]))

        print(delta)
        print(psis)
        p=max([d[T-1] for d in delta])
        I = [0 for _ in range(T)]
        #最优路径
        I[T - 1] = np.argmax([element[T - 1] for element in delta])
        for t in range(T - 2, -1, -1):
            I[t] = int(psis[int(I[t + 1])][t + 1])

        print('route:',I)
        print('probaility:',p)

HMM=HiddenMarkov(Q,V,O)
HMM.forward(A,B,PI)
HMM.backward(A,B,PI)
HMM.viterbi(A,B,PI)
