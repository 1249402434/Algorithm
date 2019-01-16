from sklearn.datasets import load_iris
import math
import numpy as np
import matplotlib.pyplot
from sklearn.model_selection import train_test_split

iris=load_iris()
data=iris.data
target=iris.target
data=data[:100,:2]
target=target[:100]
target[:50]=-1

train_x,test_x,train_y,test_y=train_test_split(data,target,test_size=0.3)

class AdaBoost:
    def __init__(self,n_estimators,learning_rate):
        self.n_estimators=n_estimators
        self.learning_rate=learning_rate

    def _init_args(self,datasets,labels):
        self.clf=[]
        self.X=datasets
        self.Y=labels
        self.M,self.N=datasets.shape
        self.weights=[1/self.M]*self.M
        self.alpha=[]

    def _G(self,features,labels):
        error=10000
        best_v=0.0
        min_value=min(features)
        max_value=max(features)
        step=(max_value-min_value+self.learning_rate)//self.learning_rate
        direct,result_array=None,None
        for i in range(1,int(step)):
            v=min_value+self.learning_rate*i
            if v not in features:
                result_array_positive=[1 if features[j]>v else -1 for j in range(self.M)]
                weights_error_positive=sum([self.weights[j] for j in range(self.M) if result_array_positive[j]!=labels[j]])

                result_array_negative=[-1 if features[j]>v else 1 for j in range(self.M)]
                weights_error_negative=sum([self.weights[j] for j in range(self.M) if result_array_negative[j]!=labels[j]])

                if weights_error_negative>weights_error_positive:
                    direct='positivate'
                    _result_array=result_array_positive
                    weights_error=weights_error_positive
                else:
                    direct='negativate'
                    _result_array=result_array_negative
                    weights_error=weights_error_negative

                if weights_error<error:
                    result_array=_result_array
                    best_v=v
                    error=weights_error

        return best_v,direct,error,result_array

    def G(self,x,v,direct):
        if direct=='positive':
            return 1 if x>v else -1
        else:
            return -1 if x>v else 1

    def _alpha(self,error):
        return 0.5*math.log((1-error)/error,math.e)

    def _Zm(self,alpha,clf_result):
        return sum([self.weights[i]*math.exp(-alpha*self.Y[i]*clf_result[i]) for i in range(self.M)])

    def _update_weights(self,zm,alpha,clf_result):
        for i in range(self.M):
            self.weights[i]=self.weights[i]*math.exp(-alpha*self.Y[i]*clf_result[i])/zm

    def fit(self,X,y):
        self._init_args(X,y)
        for epoch in range(self.n_estimators):
            best_clf_error,best_v,clf_result=10000,None,None
            final_direct,axis=None,None
            for j in range(self.N):
                features=X[:,j]
                v,direct,error,result_array=self._G(features,y)
                if best_clf_error>error:
                    best_clf_error=error
                    best_v=v
                    clf_result=result_array
                    final_direct=direct
                    axis=j

            a=self._alpha(best_clf_error)
            zm=self._Zm(a,clf_result)
            self._update_weights(zm,a,clf_result)
            self.alpha.append(a)
            self.clf.append((axis,best_v,final_direct))

            if best_clf_error==0:
                break


    def predict(self,X):
        class_result=[]
        for x in X:
            result=0
            for i in range(len(self.clf)):
                ai=self.alpha[i]
                axis,best_v,direct=self.clf[i]
                temp_result=self.G(x[axis],best_v,direct)
                result+=temp_result*ai

            if result>0:
                class_result.append(1)
            else:
                class_result.append(-1)

        return class_result

    def score(self,X_test,y_test):
        predict_result=self.predict(X_test)
        right_count=0
        length=len(X_test)

        for i in range(length):
            if y_test[i]==predict_result[i]:
                right_count+=1

        return right_count/length

adaboost=AdaBoost(30,0.1)
adaboost.fit(train_x,train_y)
print(adaboost.score(test_x,test_y))

