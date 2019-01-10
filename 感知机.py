import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class Perceptron:
    def __init__(self,alpha=0.001,max_iter=None,tol=None):
        self.W=None
        self.b=0
        self.tol=tol
        self.alpha=alpha
        self.max_iter=max_iter

    def sign(self,x):
        result=np.dot(self.W,x)+self.b
        if result>=0:
            return 1
        else:
            return -1

    def fit(self,X,y):
        length,dim=X.shape
        self.W=np.random.rand(dim)
        iter_count=0
        finished=False
        while not finished:
            error=0
            wrong_count = 0
            for i in range(length):
                symbol=self.sign(X[i])
                feature=X[i]
                label=y[i]
                if symbol*label<=0:
                    #计算误差，误分类点到直线的距离总和
                    error+=np.linalg.norm(self.W)*(-label)*symbol
                    wrong_count+=1
                    iter_count+=1
                    self.W=self.W+self.alpha*np.dot(label,feature)
                    self.b=self.b+self.alpha*label

            if wrong_count==0:
                finished=True
            if self.max_iter is not None:
                if self.max_iter<=iter_count:
                    finished=True
            if self.tol>error:
                finished=True

    def score(self,X,y):
        true_count=0
        total=len(X)
        for i in range(total):
            x=X[i]
            target=y[i]
            symbol=self.sign(x)
            if target==symbol:
                true_count+=1

        return true_count/total


iris=load_iris()
X=iris.data
y=iris.target

#首先绘制前100个样本查看分布情况
plt.scatter(X[:50,0],X[:50,1],label='0')
plt.scatter(X[50:100,0],X[50:100,1],label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()

sub_X= X[:100, :2]
sub_y= y[:100]

sub_y=[1 if i == 1 else -1 for i in sub_y]#1与-1两类

train_x,test_x,train_y,test_y=train_test_split(sub_X,sub_y,test_size=0.3,random_state=2019)

p=Perceptron(tol=0.1)
p.fit(train_x, train_y)


x_points=np.linspace(4,7,10)
y_points=-(p.W[0]*x_points+p.b)/p.W[1]#分割线公式WX+b=0，W[1]对应y轴
plt.plot(x_points,y_points)

plt.plot(sub_X[:50, 0], sub_X[:50, 1], 'bo', color='b', label='0')
plt.plot(sub_X[50:100, 0], sub_X[50:100, 1], 'bo', color='orange', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()

print(p.score(test_x,test_y))







