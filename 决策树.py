import pandas as pd
import numpy as np
import math

def create_data():
    datasets = [['青年', '否', '否', '一般', '否'],
               ['青年', '否', '否', '好', '否'],
               ['青年', '是', '否', '好', '是'],
               ['青年', '是', '是', '一般', '是'],
               ['青年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '好', '否'],
               ['中年', '是', '是', '好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '好', '是'],
               ['老年', '是', '否', '好', '是'],
               ['老年', '是', '否', '非常好', '是'],
               ['老年', '否', '否', '一般', '否'],
               ]
    labels = [u'年龄', u'有工作', u'有自己的房子', u'信贷情况', u'类别']
    # 返回数据集和每个维度的名称
    return datasets, labels

datasets,labels=create_data()
train_data=pd.DataFrame(datasets,columns=labels)

print(np.array(train_data))

#利用ID3算法生成决策树

#节点类
class Node:
    '''
    若此节点不可再分，root取True，而且label不为None
    可以再分root=False，label=None，feature_name和feature不为空
    '''
    def __init__(self,root=True,label=None,feature_name=None,feature=None):
        self.root=root
        self.feature_name=feature_name
        self.feature=feature
        self.label=label
        self.tree={}
        self.result={'label':label,'feature_name':feature_name,'tree':self.tree}

    def __repr__(self):
        return '{}'.format(self.result)

    #此节点可分裂出多个节点，不一定是二叉树
    def add_node(self,index,node):
        self.tree[index]=node

    def predict(self,features):
        if self.root==True:
            return self.label
        return self.tree[features[self.feature]].predict(features)

class DecisionTree:
    def __init__(self,epsilon=0.1):
        self.epsilon=epsilon    #分裂阈值
        self._tree={}

    def entropy(self,p):
        return (-p) * math.log2(p)

    # 经验熵
    def calc_entropy(self,datasets):
        length = len(datasets)
        label_dict = {}
        for i in range(length):
            label = datasets[i][-1]
            #统计每个类别的样本数量
            label_dict[label] = label_dict.get(label, 0) + 1

        total = sum(label_dict.values())
        experience_entropy = 0
        for key in label_dict.keys():
            p = label_dict.get(key) / total
            experience_entropy += self.entropy(p)

        return experience_entropy

    # 经验条件熵
    def condition_entropy(self, datasets, cond):
        feature_sets = {}
        length = len(datasets)
        for i in range(length):
            #把当前特征的特征值相同的样本放到同一个列表中
            feature = datasets[i][cond]
            if feature not in feature_sets:
                feature_sets[feature] = []

            feature_sets[feature].append(datasets[i])
        return sum([len(data) / length * self.calc_entropy(data) for data in feature_sets.values()])

    def info_gain(self,entropy, cond_entropy):
        return entropy - cond_entropy

    #计算并选取信息增益最大的特征
    def info_gain_train(self,datasets):
        feature_count = len(datasets[0]) - 1
        entropy = self.calc_entropy(datasets)
        feature_score = []
        for c in range(feature_count):
            cond_entropy = self.condition_entropy(datasets, c)
            gain = self.info_gain(entropy, cond_entropy)
            feature_score.append((c, gain))

        best_ = max(feature_score, key=lambda x: x[1])
        return best_

    def train(self,data):
        train_x,train_y,feature_names=data.iloc[:,:-1],data.iloc[:,-1],data.columns[:-1]
        #若D中所有实例属于同一类Ck，则T为单节点树，并将类Ck作为该节点的类标记，返回T
        if len(train_y.value_counts())==1:
            return Node(root=True,label=train_y.iloc[0])
        #若A为空，则T为单节点树，并将D中实例数最大的类Ck作为该节点的类标记，返回T
        if len(feature_names)==0:
            return Node(root=True,label=train_y.value_counts().sort_values(ascending=False).index[0])
        #选取信息增益最大的特征Ag
        max_feature,score=self.info_gain_train(np.array(data))
        max_feature_name=feature_names[max_feature]
        #如果信息增益小于阈值epsilon，则T为单节点树，并将D中实例数最大的类Ck作为该节点的类标记，返回T
        if score<self.epsilon:
            return Node(root=True,label=train_y.value_counts.sort_values(ascending=False).index[0])

        '''
        否则，对Ag的每一可能值ai，依Ag=ai将D分割为若干非空子集Di，将Di中实例最大的类作为标记，构建子节点
        由节点及其子节点构成树T，返回T
        '''
        node_tree=Node(root=False,label=None,feature_name=max_feature_name,feature=max_feature)

        '''
        对第i个子节点，以Di为训练集，以A-{Ag}为特征集，递归调用
        '''
        feature_list=data[max_feature_name].value_counts().index
        for f in feature_list:
            child_train=data.loc[data[max_feature_name]==f].drop([max_feature_name],axis=1)
            child=self.train(child_train)
            node_tree.add_node(f,child)

        return node_tree

    def fit(self,data):
        self._tree=self.train(data)
        return self._tree

    def predict(self,X):
        return self._tree.predict(X)

decisiontree=DecisionTree()
generate_tree=decisiontree.fit(train_data)
print(generate_tree)
print(generate_tree.predict(['老年', '是', '否', '一般']))

#决策树的可视化
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

iris=load_iris()
X, y = iris.data,iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
#命令行输入  dot -Tpdf mytree.dot -o mytree.pdf 可转成pdf文件查看
#value字段代表了每个类样本的具体数量
tree_pic = export_graphviz(clf, out_file="mytree.dot",feature_names=iris.feature_names,class_names=iris.target_names)








