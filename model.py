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

print(featureSet.groupby(featureSet['label']).size())


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
print(winner)

clf = model[winner]
res = clf.predict(X)
mt = confusion_matrix(y, res)
print("False positive rate : %f %%" % ((mt[0][1] / float(sum(mt[0])))*100))
print('False negative rate : %f %%' % ( (mt[1][0] / float(sum(mt[1]))*100)))

