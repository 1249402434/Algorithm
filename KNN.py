import math
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt
from collections import namedtuple
from time import clock
from random import random

class ValueError(Exception):
    def __init__(self,log):
        self.log=log

    def __str__(self):
        print(self.log)

def L(X1,X2,p=2):
    length=len(X1)
    if length==len(X2) and length>=1:
        sum=0
        for i in range(length):
            sum+=math.pow(X1[i]-X2[i],p)
        return math.pow(sum,1/p)
    else:
        raise ValueError('Value Error!')

class KNN:
    def __init__(self,x_train,y_train,k=3,p=2):
        self.neighbors=k
        self.p=p
        self.x_train=x_train
        self.y_train=y_train

    def predict(self,X):
        self.neighbors_dist=[]
        #首先取前k个数据点作为距离最近的点，以后再进行更新
        for i in range(self.neighbors):
            dist=np.linalg.norm(X-self.x_train[i],ord=self.p)
            self.neighbors_dist.append((dist,self.y_train[i]))

        for i in range(self.neighbors,len(self.x_train)):
            dist=np.linalg.norm(X-self.x_train[i],ord=self.p)
            max_index=self.neighbors_dist.index(max(self.neighbors_dist,key=lambda x:x[0]))
            if dist<self.neighbors_dist[max_index][0]:
                self.neighbors_dist[max_index]=(dist,self.y_train[i])

        class_value=[neighbor[1] for neighbor in self.neighbors_dist]
        #自动统计列表中各值的对应数目，是字典的子类
        counter=Counter(class_value)
        result=sorted(counter.items(),key=lambda x:x[1],reverse=True)[0]
        return result[0]

    def score(self,X,y):
        count=0
        for x_,y_ in zip(X,y):
            predict_result=self.predict(x_)
            if predict_result==y_:
                count+=1

        return count/len(X)

iris=load_iris()
data=iris.data
target=iris.target

X=data[:100,:2]
y=target[:100]

plt.scatter(X[:50,0],X[:50,1],label='0')
plt.scatter(X[50:100,0],X[50:100,1],label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=.3,random_state=2019)

knn=KNN(x_train,y_train)
print(knn.score(x_test,y_test))

test_point=[5.0,3.2]
test_target=knn.predict(test_point)

plt.scatter(X[:50,0],X[:50,1],label='0')
plt.scatter(X[50:100,0],X[50:100,1],label='1')
plt.scatter(test_point[0],test_point[1],color='r',label=str(test_target))
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()


#kd树
class Node:
    def __init__(self,element,split,left,right):
        self.element=element
        self.split=split    #选取的切分轴
        self.left=left
        self.right=right


class KDTree:
    def __init__(self,data):
        if data==None:
            print('data is null')

        dim=len(data[0])
        def CreateNode(split,dataset):
            if not dataset:
                return None

            dataset.sort(key=lambda x:x[split])
            pos=len(dataset)//2
            median=dataset[pos]#当前节点的元素
            split_next=(split+1)%dim#更新划分轴

            return Node(median,split,
                        CreateNode(split_next,dataset[:pos]),
                        CreateNode(split_next,dataset[pos+1:]))#注意左子树和右子树都不要包含pos位置

        self.root=CreateNode(0,data)


def preorder(root):
    print(root.element)
    if root.left:
        preorder(root.left)
    if root.right:
        preorder(root.right)

d=[(2,3),(5,4),(9,6),(4,7),(8,1),(7,2)]
tree=KDTree(d)
preorder(tree.root)

#kd树的搜索
#命名元祖，后面以空格隔开的三个元素可以看作是该对象的成员变量
result=namedtuple('result','nearest_point nearest_dist nodes_visit')
def find_nearest_point(tree,target):
    k=len(target)#目标点维度
    def recursion_visit(node,point,max_dist):
        if node is None:
            return result([0]*k,float('inf'),0)

        nodes_visited=1
        s=node.split#当前节点的划分轴
        element=node.element#当前节点存储的数据
        #以指定的划分轴去比较当前节点的值与目标值，然后递归搜索，直到叶子节点
        if point[s]<=element[s]:
            nearer_node=node.left
            futher_node=node.right
        else:
            nearer_node=node.right
            futher_node=node.left

        temp1=recursion_visit(nearer_node,point,max_dist)

        #假设包含目标点的叶子节点为距离目标点最近的点
        nearest=temp1.nearest_point
        dist=temp1.nearest_dist
        nodes_visited+=temp1.nodes_visit
        #更新搜索半径
        if max_dist>dist:
            max_dist=dist

        #判断目标点的超球体与此叶子节点的父节点的超矩形区域是否相交，不相交直接返回
        temp_dist=abs(element[s]-point[s])
        if temp_dist>max_dist:
            return result(nearest,dist,nodes_visited)

        temp_dist=math.sqrt(sum((p1-p2)**2 for p1,p2 in zip(element,point)))
        #相交且距离父节点更近，更新状态(搜索半径)，也有可能不更新，就是直接以当前半径去搜索父节点的右孩子
        if temp_dist<dist:
            nearest=element
            dist=temp_dist
            max_dist=dist

        #相交判断父节点的右孩子中是否存在距离更近的点
        temp2=recursion_visit(futher_node,point,max_dist)
        nodes_visited+=temp2.nodes_visit
        if temp2.nearest_dist<dist:
            nearest=temp2.nearest_point
            dist=temp2.nearest_dist

        return result(nearest,dist,nodes_visited)

    return recursion_visit(tree.root,target,float('inf'))


ret = find_nearest_point(tree, [3,4.5])
print(ret)

# 产生一个k维随机向量，每维分量值在0~1之间
def random_point(k):
    return [random() for _ in range(k)]

# 产生n个k维随机向量
def random_points(k, n):
    return [random_point(k) for _ in range(n)]

N = 400000
# clock()函数以浮点数计算的秒数返回当前的CPU时间。用来衡量不同程序的耗时，比time.time()更有用
t0 = clock()
kd2 = KDTree(random_points(3, N))            # 构建包含四十万个3维空间样本点的kd树
ret2 = find_nearest_point(kd2, [0.1,0.5,0.8])      # 四十万个样本点中寻找离目标最近的点
t1 = clock()
print ("time: ",t1-t0, "s")
print (ret2)
