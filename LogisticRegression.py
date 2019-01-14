import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
import math
from sklearn.model_selection import train_test_split

iris=load_iris()
print(iris.feature_names)
data=iris.data
target=iris.target

data=data[:100,:2]
target=target[:100]

train_x,test_x,train_y,test_y=train_test_split(data,target,test_size=0.3,random_state=2019)

class LogisticRegression:
    def __init__(self,tol=1e-4,max_iter=500,epsilon=0.5,lr=0.01):
        self.tol=tol
        self.max_iter=max_iter
        self.lr=lr
        self.W=None
        self.epsilon=epsilon

    def data_matrix(self,data):
        data_mat=[]
        for d in data:
            '''为原始数据增加一维，方便与多项式中的W0相乘，代替b'''
            data_mat.append([1.0,*d])
        return data_mat

    def sigmoid(self,x):
        return 1/(1+math.exp(-x))

    '''对数损失函数'''
    def loss_function(self,y_hat,y):
        return -(y*math.log2(y_hat)+(1-y)*math.log2(1-y_hat))

    def fit(self,X,y):
        num,feature=X.shape
        self.W=np.zeros((feature+1,1),dtype=np.float32)
        data_mat=self.data_matrix(X)
        for i in range(self.max_iter):
            error=0
            for j in range(num):
                y_hat=self.sigmoid(np.dot(data_mat[j],self.W))
                error+=self.loss_function(y_hat,y[j])
                '''损失函数关于权重的偏导数'''
                self.W-=self.lr*(y_hat-y[j])*np.transpose([data_mat[j]])
            if error<=self.tol:
                break

    def predict(self,X):
        num=X.shape[0]
        data_mat=self.data_matrix(X)
        result=[]
        for i in range(num):
            y_hat=self.sigmoid(np.dot(data_mat[i],self.W))
            if y_hat>=self.epsilon:
                result.append(1)
            else:
                result.append(0)

        return result

    def score(self,X_test,y_test):
        length=len(X_test)
        result=self.predict(X_test)
        right_count=0
        for i in range(length):
            if result[i]==y_test[i]:
                right_count+=1

        return right_count/length

model=LogisticRegression()
model.fit(train_x,train_y)
print(model.score(test_x,test_y))

plt.scatter(data[:50,0],data[:50,1],label='0')
plt.scatter(data[50:100,0],data[50:100,1],label='1')
x_points = np.arange(2, 5)
y_ = -(model.W[2]*x_points + model.W[0])/model.W[1]
plt.plot(x_points, y_)
plt.legend()
plt.show()