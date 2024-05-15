# -*- coding: utf-8 -*-
"""Xgboost_2017311362박경원

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-N2OQXXUqimWCPepLTJxUZocT5KR1-Xf
"""

#Importing dataset from sklearn

from sklearn import datasets
from sklearn import metrics
import numpy as np
import pandas as pd
train=pd.read_csv("./train.csv")
test=pd.read_csv("./test.csv")

train.isna().sum()

import matplotlib.pyplot as plt
columns=train.columns
train.boxplot(column=['emotion_angry_mean', 'emotion_disgust_mean', 
                      'emotion_fear_mean', 'emotion_happy_mean', 'emotion_sad_mean', 'emotion_surprise_mean'])

#라벨과 각 피쳐의 상관관계 구하기
import scipy.stats as stats
p_value={}
p_value_real={}
p_values_value=[]
for i,o in enumerate(train.columns) :
  X=train['label']
  Y=train[o]
  p_values_value.append(stats.pearsonr(X,Y)[1])
  p_value_real[o] = stats.pearsonr(X,Y)[1]

p_values_value = sorted(p_values_value, reverse=True)
#상관관계가 높은 상위 50개의 features 들을 골라낸다.
p_values_value=p_values_value[0:383]
# print(p_values_value)
for i,o in enumerate(train.columns) :
  for a in p_values_value :
    if p_value_real[o] == a :
        p_value[o]= [a]
train_10_df=pd.DataFrame(p_value)

print(train_10_df)


X= train.loc[0:399,train_10_df.columns]
y= train.iloc[0:400,383]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=31)
X_train, X_test, y_train, y_test=np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
train_x = ss.fit_transform(X_train)
test_x = ss.transform(X_test)

X= train.loc[0:399,train_10_df.columns]
X.shape

#싸이킷런으로 RandomForest 구현


from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

clf_RF=RandomForestClassifier()
clf_XGB = XGBClassifier()
clf_NB= GaussianNB()
clf_SVM=SVC()
clf_DT=DecisionTreeRegressor()
clf_GBC=GradientBoostingClassifier()
clf_LGB=LGBMClassifier()
clf_LR=LogisticRegression()

clf_XGB.fit(X_train, y_train)
y_pred = clf_XGB.predict(X_test)

print(metrics.classification_report(y_test, y_pred))

# 이름과 학번을 작성해주세요
__author__ = "박경원"
__id__ = "2017311362"

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor


class XG_Boosting():
    def __init__(self):
        """
        파라미터 설정(학습률, estimator)
        """

        self.author = __author__
        self.id = __id__

        self.boosting_N =20 #Estimator
        self.learning_rate = 0.3 # 0.01~0.3 싸이킷런 라이브러리의 default learning rate 0.3
        self.lamda=1

        self.weak_classifier = []
        self.score = None


    def sigmoid(self, x):
        """
        모든 데이터가 0에서 1사이의 값을 가지도록 설정해줍니다.
        모든 x에 대한 sigmoid 결과 값 도출
        """
         
        return 1 / (1 + np.exp(-x))       # 시그모이드 함수 공식 : 1/(1+exp(-x)) = exp(x)/(exp(x)+1)

    def Gradient(self, y, x):
        """
        Gradient를 연산하는 함수이며, loss를 근사하기 위해 사용됩니다.
        """
        return (y - self.sigmoid(x))

    def Hessian(self, x):
        """
         Hessian을 연산하는 함수로, loss의 gradient 값을 계산합니다.
        """
        return (self.sigmoid(x) * (1 - self.sigmoid(x)))

    def fit(self, X_train, y_train):
        """
        본 함수는 X_train, y_train를 활용하여 훈련하는 과정을 코딩하는 부분입니다.
        설정한 estimator인 self.boosting_N만큼 훈련을 반복합니다.
        훈련을 할 때 마다 임의의 decision tree 생성 한 후, 설정한 가중치를 기반으로 모델을 학습하고,
        학습된 Classifier는 self.weak_classifier 리스트에 저장되며,
        해당 Classifier의 예측 값은 self.learning rate를 반영하여 기존의 self.score와 합산
        """
        a = np.log(sum(y_train[y_train ==1]))/(1-sum(y_train[y_train ==1]))
        self.score=np.zeros(X_train.shape[0]) 
        self.score=self.score+a
        ''' 스코어의 초기 값 구하기'''
        grad = self.Gradient(y_train, self.score)
        hess = self.Hessian(self.score) 
        sample_weights = grad/(1+hess)
        #[1/len(y_train) for i in range(len(y_train))] # 임의의 가중치 할당
        '''가중치의 초기 값 구하기'''
        y_new = y_train
        

        for i in range(self.boosting_N):
            '''
            임의의 decision tree 생성 한 후, 설정한 가중치를 기반으로 모델 학습을 진행
            학습된 Classifier는 self.weak_classifier 리스트에 저장
            '''
            new_dt = DecisionTreeRegressor()
            new_dt.fit(X_train, y_new, sample_weight=sample_weights) 
            self.weak_classifier.append(new_dt) 
            
            '''
            DT모델의 X_train 데이터에 대한 prediction score를 도출
            앞서 만든 DecisionTreeRegressor() 를 이용하여 도출 
            new_score에 learning rate를 반영하여 self.score를 업데이트
            self.score에 learning_rate를 반영한 값을 더한다.      
            gradient와 hessian을 반영하여 y_new와 가중치 업데이트
            '''
            new_score = new_dt.predict(X_train)      
            self.score = self.score + (self.learning_rate) * new_score   
            y_new = grad/hess
            sample_weights = hess

    def predict(self, X_test):
        """
        이 함수는 X_test가 주어졌을 때, fit을 통해 계산된 classifier를
        각 모델의 예측값과 합산하여 최종 도출
        """
        a = np.log(sum(y_train[y_train ==1]))/(1-sum(y_train[y_train ==1]))
        pred=np.zeros(X_test.shape[0]) 
        pred=pred+a
       
        for w_classifier in self.weak_classifier:
            pred = pred + (1+self.learning_rate)*w_classifier.predict(X_test) 
            # learning_rate를 곱 연산한 후 predict 값을 pred에 더하여 값을 구한다.

        probas = self.sigmoid(np.full((X_test.shape[0], 1), 1).flatten().astype('float64') + pred)
        pred_label = np.where(probas >  np.mean(probas),1,0) #np조건문을 사용하여 평균보다 크면 1, 작으면 0으로 할당 
        return pred_label

clf = XG_Boosting()

clf.fit(X_train, y_train)
print(clf.predict(X_test))
y_pred = clf.predict(X_test)

print(metrics.classification_report(y_test, y_pred))

test_set=test.iloc[:,1:384]
test_set.shape

clf = XG_Boosting()
clf.fit(X_train, y_train)
test_set=test.loc[:,train_10_df.columns]
y_pred = clf.predict(test_set)

new_df=pd.DataFrame({'id':test['id'],'label':y_pred})
new_df.to_csv("PKW_submission.csv", index = False)