'''
此SVM实现采用的是SMO算法
'''

from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split

iris=load_iris()
data=iris.data
target=iris.target
data=data[:100,:2]
target=target[:100]
target[:50]=-1


class SVM:
    def __init__(self,max_iter,kernel='linear'):
        self.max_iter=max_iter
        self.kernel=kernel

    def _init_args(self,features,labels):
        self.m,self.n=features.shape
        self.X=features
        self.Y=labels
        #存储alpha的列表
        self.alpha=np.ones(self.m)
        self.C=1.0
        self.b=0.0
        self.E=[self._E(i) for i in range(self.m)]

    #g(x)
    def _g(self,i):
        r=self.b
        for j in range(self.m):
            r+=self.alpha[j]*self.Y[j]*self._kernel(self.X[j],self.X[i])
        return r

    #线性核函数和多项式核函数
    def _kernel(self,xi,x):
        if self.kernel=='linear':
            return sum([xi[k]*x[k] for k in range(self.n)])
        elif self.kernel=='poly':
            return (sum([xi[k]*x[k] for k in range(self.n)])+1)**2

    #是否满足KKT条件
    def _KKT(self,i):
        yi=self.Y[i]
        gi=self._g(i)
        res=yi*gi
        if self.alpha[i]==0:
            return res>=1
        elif self.alpha[i]>0 and self.alpha[i]<self.C:
            return res==1
        else:
            return res<=1

    def _E(self,i):
        return self._g(i)-self.Y[i]

    def _choose_alpha(self):
        #先遍历在间隔边界上的点0<alpha<C，所以索引存放应按照以下顺序
        index_belowC = [i for i in range(self.m) if self.alpha[i] > 0 and self.alpha[i] < self.C]
        remain_index = [i for i in range(self.m) if i not in index_belowC]
        index_belowC.extend(remain_index)

        for i in index_belowC:
            if self._KKT(i):
                continue
            '''
            此函数要选择两个alpha，通过KKT条件可先选取第一个
            然后凭借E1-E2的绝对值最大的标准选取第二个alpha    
            '''
            E1 = self.E[i]
            if E1>=0:
                j=min(range(self.m),key=lambda x:self.E[x])
            else:
                j=max(range(self.m),key=lambda x:self.E[x])

            return i,j

    #对alpha值更新函数
    def _compare(self,alpha_new_unc,H,L):
        if alpha_new_unc>H:
            return H
        elif alpha_new_unc<L:
            return L
        else:
            return alpha_new_unc

    def fit(self,X,y):
        self._init_args(X,y)
        for iter in range(self.max_iter):
            i,j=self._choose_alpha()
            alpha1=self.alpha[i]
            alpha2=self.alpha[j]

            '''
            根据约束条件确定alpha的取值范围
            '''
            if self.Y[i]==self.Y[j]:
                L=max(0,self.alpha[i]+self.alpha[j]-self.C)
                H=min(self.C,self.alpha[i]+self.alpha[j])
            else:
                L=max(0,self.alpha[j]-self.alpha[i])
                H=min(self.C,self.C+self.alpha[j]-self.alpha[i])

            E1=self.E[i]
            E2=self.E[j]
            eta=self._kernel(self.X[i],self.X[i])+self._kernel(self.X[j],self.X[j])-2*self._kernel(self.X[i],self.X[j])
            if eta<=0:
                continue

            alpha_new_unc2=alpha2+self.Y[j]*(E1-E2)/eta
            alpha_new2=self._compare(alpha_new_unc2,H,L)

            alpha_new1=alpha1+self.Y[i]*self.Y[j]*(alpha2-alpha_new2)

            b1_new=-E1-self.Y[i]*self._kernel(self.X[i],self.X[i])*(alpha_new1-alpha1)\
                   -self.Y[j]*self._kernel(self.X[j],self.X[i])*(alpha_new2-alpha2)+self.b

            b2_new=-E2-self.Y[i]*self._kernel(self.X[i],self.X[j])*(alpha_new1-alpha1)\
                   -self.Y[j]*self._kernel(self.X[j],self.X[j])*(alpha_new2-alpha2)+self.b

            if 0<alpha_new1<self.C:
                b_new=b1_new
            elif 0<alpha_new2<self.C:
                b_new=b2_new
            else:
                b_new=(b1_new+b2_new)/2

            #对b,alpha,E值进行更新
            self.b=b_new
            self.alpha[i]=alpha_new1
            self.alpha[j]=alpha_new2
            self.E[i]=self._E(i)
            self.E[j]=self._E(j)

        return "finish"

    def predict(self,x):
        r=self.b
        for i in range(self.m):
            r+=self.alpha[i]*self.Y[i]*self._kernel(self.X[i],x)

        return 1 if r>0 else -1

    def score(self,X_test,y_test):
        right_count=0
        length=len(X_test)
        for i in range(length):
            temp=self.predict(X_test[i])
            if temp==y_test[i]:
                right_count+=1

        return right_count/length

for i in range(10):
    train_x,test_x,train_y,test_y=train_test_split(data,target,test_size=0.3)
    svm=SVM(300)
    svm.fit(train_x,train_y)
    print(svm.score(test_x,test_y))




