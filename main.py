from __future__ import division
import os
import sys
import re
import matplotlib
import pandas as pd
import numpy as np
from os.path import splitext
import ipaddress as ip
import tldextract
import whois
import datetime
from urllib.parse import urlparse

df = pd.read_csv("dataset.csv")

df = df.sample(frac=1).reset_index(drop=True)
df.head()
print("Total no. of entries : ", len(df)) 
Suspicious_TLD=['zip','cricket','link','work','party','gq','kim','country','science','tk']
Suspicious_Domain=['luckytime.co.kr','mattfoll.eu.interia.pl','trafficholder.com','dl.baixaki.com.br','bembed.redtube.comr','tags.expo9.exponential.com','deepspacer.com','funad.co.kr','trafficconverter.biz']

def countdots(url):  
    return url.count('.')
def countdelim(url):
    count = 0
    delim=[';','_','?','=','&']
    for each in url:
        if each in delim:
            count = count + 1
    return count
import ipaddress as ip 

def isip(uri):
    try:
        if ip.ip_address(uri):
            return 1
    except:
        return 0


def isPresentHyphen(url):
    return url.count('-')

def isPresentAt(url):
    return url.count('@')


def isPresentDSlash(url):
    return url.count('//')

def countSubDir(url):
    return url.count('/')
def get_ext(url):
    
    root, ext = splitext(url)
    return ext
def countSubDomain(subdomain):
    if not subdomain:
        return 0
    else:
        return len(subdomain.split('.'))


def countQueries(query):
    if not query:
        return 0
    else:
        return len(query.split('&'))

featureSet = pd.DataFrame(columns=('url','no of dots','presence of hyphen','len of url','presence of at',\
'presence of double slash','no of subdir','no of subdomain','len of domain','no of queries','is IP','presence of Suspicious_TLD',\
'presence of suspicious domain','label'))

from urllib.parse import urlparse
import tldextract
def getFeatures(url, label): 
    result = []
    url = str(url)
    result.append(url)
    path = urlparse(url)
    ext = tldextract.extract(url)
    result.append(countdots(ext.subdomain))
    result.append(isPresentHyphen(path.netloc))
    result.append(len(url))
    result.append(isPresentAt(path.netloc))
    result.append(isPresentDSlash(path.path))
    result.append(countSubDir(path.path))
    result.append(countSubDomain(ext.subdomain))
    result.append(len(path.netloc))
    result.append(len(path.query))
    result.append(isip(ext.domain))
    result.append(1 if ext.suffix in Suspicious_TLD else 0)
    result.append(1 if '.'.join(ext[1:]) in Suspicious_Domain else 0 )

    result.append(str(label))
    return result

for i in range(len(df)):
    features = getFeatures(df["URL"].loc[i], df["Lable"].loc[i])    
    featureSet.loc[i] = features
print("---------------------------------------------------------")
print(" ")
print("Final FeatureSet : ",featureSet.head())
print(" ")
print("---------------------------------------------------------")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle as pkl

sns.set(style="darkgrid")
sns.distplot(featureSet[featureSet['label']=='0']['len of url'],color='green',label='Benign URLs')
sns.distplot(featureSet[featureSet['label']=='1']['len of url'],color='red',label='Phishing URLs')
import matplotlib.pyplot as plt
plt.title('Url Length Distribution')
plt.legend(loc='upper right')
plt.xlabel('Length of URL')
plt.show()

import sklearn.ensemble as ek
from sklearn import tree, linear_model
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import cross_validate,cross_val_score
#from sklearn import cross_validation
from sklearn.feature_selection import SelectFromModel
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn import svm
from sklearn.linear_model import LogisticRegression
print(" ")
print(featureSet.groupby(featureSet['label']).size())
print(" ")


X = featureSet.drop(['url','label'],axis=1).values
y = featureSet['label'].values

model = { "DecisionTree":tree.DecisionTreeClassifier(max_depth=10),
         "RandomForest":ek.RandomForestClassifier(n_estimators=50),
         "Adaboost":ek.AdaBoostClassifier(n_estimators=50),
         "GradientBoosting":ek.GradientBoostingClassifier(n_estimators=50),
         "GNB":GaussianNB(),
         "LogisticRegression":LogisticRegression()   
}
X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.2)

results = {}
for algo in model:
    clf = model[algo]
    clf.fit(X_train,y_train)
    score = clf.score(X_test,y_test)
    print ("%s : %s " %(algo, score))
    results[algo] = score

winner = max(results, key=results.get)
print(" ")
print("Model with highest accuracy is : ",winner)
print(" ")

clf = model[winner]
res = clf.predict(X)
mt = confusion_matrix(y, res)
print("-----------------------------------------------------------------------")
print(" ")
print("False positive rate : %f %%" % ((mt[0][1] / float(sum(mt[0])))*100))
print('False negative rate : %f %%' % ( (mt[1][0] / float(sum(mt[1]))*100)))
result = pd.DataFrame(columns=('url','no of dots','presence of hyphen','len of url','presence of at',\
'presence of double slash','no of subdir','no of subdomain','len of domain','no of queries','is IP','presence of Suspicious_TLD',\
'presence of suspicious domain','label'))
print(" ")
print("-----------------------------------------------------------------------")
web = input("Enter the website :")

results = getFeatures(web, '1')
result.loc[0] = results
result = result.drop(['url','label'],axis=1).values
print(" ")
print("Final Result : ",clf.predict(result))
