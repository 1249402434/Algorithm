'''
以下是基于高斯判别模型的贝叶斯实现
能够处理特征值是连续值的情况，与一般处理离散属性值的
多项式模型不同
'''
import math
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter

class NaiveBayes:
    def __init__(self):
        self.model=None
        self.class_proba={}#p(Y)

    def mean(self,X):
        return sum(X)/len(X)

    def stdev(self,X):
        avg=self.mean(X)
        return math.sqrt(sum([pow(x-avg,2) for x in X])/len(X))

    #高斯公式
    def gaussian(self,x,mean,stdev):
        exponent=math.exp(-pow((x-mean),2)/(2*pow(stdev,2)))
        return 1/(math.sqrt(2*math.pi)*stdev)*exponent

    #对于每个特征计算其均值与标准差
    def summarize(self,X):
        summaries=[(self.mean(x),self.stdev(x)) for x in zip(*X)]
        return summaries

    def fit(self,X,y):
        labels=list(set(y))
        data={label:[] for label in labels}
        for feature,label in zip(X,y):
            data[label].append(feature)

        #计算并存放每一个类的每一属性的均值与标准差
        self.model={label:self.summarize(feature) for label,feature in data.items()}
        c=Counter(y)
        for label in labels:
            self.class_proba[label]=c[label]/len(y)

    def calc_proba(self,test_data):
        possibilities = {}
        length = len(test_data)
        for label, value in self.model.items():
            class_proba = self.class_proba[label]
            for i in range(length):
                mean, stdev = value[i]
                class_proba *= self.gaussian(test_data[i], mean, stdev)
            possibilities[label] = class_proba

        return possibilities

    def predict(self,test_data):
        possibilities=self.calc_proba(test_data)
        return sorted(possibilities.items(),key=lambda x:x[1])[-1][0]

    def score(self,test_feature,test_label):
        length=len(test_label)
        right_count=0
        for i in range(length):
            predict_result=self.predict(test_feature[i])
            if predict_result==test_label[i]:
                right_count+=1

        return right_count/length

iris=load_iris()
data=iris.data
target=iris.target
data=data[:100,:]
target=target[:100]

train_x,test_x,train_y,test_y=train_test_split(data,target,test_size=.3,random_state=2019)
bayes=NaiveBayes()
bayes.fit(train_x,train_y)
print(bayes.predict([4.4,  3.2,  1.3,  0.2]))

print(bayes.score(test_x,test_y))


