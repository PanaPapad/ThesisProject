
import matplotlib_inline

import os
import pandas as pd
import csv
import simpy
import re
import time

from pandas import DataFrame
import threading
import multiprocessing
import logging
import math
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
import warnings
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns

from sklearn.model_selection import train_test_split,cross_val_score


from xgboost import XGBClassifier


df1=pd.read_csv(r'C:/Users/User/Desktop/work/UCY/datasets_analyzed/IoT/analysis/#1/1sec/experiment no DoS Binary/METHOD B/new_training.csv')
df2=pd.read_csv(r'C:/Users/User/Desktop/work/UCY/datasets_analyzed/IoT/analysis/#1/1sec/experiment no DoS Binary/METHOD B/evaluation_files/new_evaluation_DOS_5.csv')



path_to_file="C:/Users/User/Desktop/work/UCY/datasets_analyzed/IoT/analysis/#1/1sec/experiment no DoS Binary/METHOD B/RESULTS/"


from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix, classification_report
f = open(path_to_file+"ML_results_python.txt", "a")

########################################## MACHINE LEARNING CLASSIFICATION ##################################################################
####################################### data pre processing #################################

normal_train = df1.columns
attack_train= df2.columns

## column in normal and not on the attack ######

index1=normal_train.difference(attack_train)
print(index1)

X_train=df1.iloc[:,1:len(df1.columns)] #get all the features
X_test=df2.iloc[:,1:len(df2.columns)]
y_train=df1["Class"]
y_test=df2["Class"]
#print(X_train)
print(X_train)



###################################### after spliting perform scaling ########################
sc_x = StandardScaler()
xtrain = sc_x.fit_transform(X_train)
xtest = sc_x.transform(X_test) ##### use transform because we want the same scaler as in the training set


#pd.DataFrame(xtest).to_csv("../test.csv")
#pd.DataFrame(xtrain).to_csv("../train.csv")


###################################### apply the ML algorithm and grid search if needed ##################################




classifier=LogisticRegression(max_iter=12000,C=1e-05,penalty="none",random_state=0)



print("################################################# K fold cross validation results ###############################################")
_scoring = ['balanced_accuracy','f1','roc_auc']
...
# define search space


...
# define search space
space = dict()
#space['solver'] = [ 'lbfgs', 'liblinear']
space['penalty'] = ['none',  'l2']
space['C'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]

LRparam_grid =[
                {'solver': ['newton-cg'],'penalty': ['l2','none'],'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100] },
    {'solver': ['lbfgs'],'penalty': ['l2','none'],'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]},
{'solver': ['sag'],'penalty': ['l2','none'],'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]},
{'solver': ['saga'],'penalty': ["elasticnet", "l1", "l2", "none"],'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]},

]

cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
score= cross_validate(classifier,xtrain,y_train,cv=cv ,scoring=_scoring,return_train_score=True)
classifier.fit(xtrain,y_train)
scores=pd.DataFrame(score)
print(scores)
print("Mean of parameters {} %".format(scores.mean()*100))
train_LG_cross_result_f1=scores['train_f1'].mean()*100
test_LG_cross_result_f1=scores['test_f1'].mean()*100
train_LG_cross_result_roc=scores['train_roc_auc'].mean()*100
test_LG_cross_result_roc=scores['test_roc_auc'].mean()*100
train_LG_cross_result_BA=scores['train_balanced_accuracy'].mean()*100
test_LG_cross_result_BA=scores['test_balanced_accuracy'].mean()*100
y_final=classifier.predict(xtest)
print("Predictions LOGISTIC REGRESSION",y_final)

print(confusion_matrix(y_test,y_final))
accuracy_LR=metrics.accuracy_score(y_test, y_final)
f1_score_LR=metrics.f1_score(y_test, y_final)
MCC_LR=metrics.matthews_corrcoef(y_test, y_final)
kappa_LG=metrics.cohen_kappa_score(y_test,y_final)
print("Accuracy:",metrics.accuracy_score(y_test, y_final))
print("Balanced Accuracy:",metrics.balanced_accuracy_score(y_test, y_final))
print("Precision:",metrics.precision_score(y_test, y_final))
print("Recall:",metrics.recall_score(y_test, y_final))
print("F1-score:",metrics.f1_score(y_test, y_final))
print("MCC:",metrics.matthews_corrcoef(y_test, y_final))
print('Kappa score:',metrics.cohen_kappa_score(y_test,y_final))
y_pred_proba = classifier.predict_proba(xtest)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
cm=confusion_matrix(y_test,y_final)
FPR_LG=cm[0][1]/ (cm[0][1]+cm[0][0])
auc = metrics.roc_auc_score(y_test, y_pred_proba)
print("AUC:",auc)
######################################### Result after Grid Search ########################
grid_search_cv = GridSearchCV(classifier, space, refit=True, verbose=3,cv=cv,scoring="f1")
result=grid_search_cv.fit(xtrain,y_train)

best_LG=result.best_estimator_
print('-------------------------------GRID SEARCH RESULTS LOGISTIC REGRESSION--------------------------------')
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)
y_pred=best_LG.predict(xtest)



print(confusion_matrix(y_test,y_pred))

print("Predictions LOGISTIC REGRESSION",y_pred)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Balanced Accuracy:",metrics.balanced_accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("F1-score:",metrics.f1_score(y_test, y_pred))
print("MCC:",metrics.matthews_corrcoef(y_test, y_pred))
print('Kappa score:',metrics.cohen_kappa_score(y_test,y_pred))
y_pred_proba_LG = best_LG.predict_proba(xtest)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba_LG)
auc_LG = metrics.roc_auc_score(y_test, y_pred_proba_LG)
print("AUC:",auc_LG)



f.write("---------------------------------------- LOGSTIC REGRESSION ----------------------------------------------------------\n")
f.write(str(confusion_matrix(y_test,y_final)))
f.write("\nAccuracy:"+ str(metrics.accuracy_score(y_test, y_final)))
f.write("\nBalanced Accuracy:"+str(metrics.balanced_accuracy_score(y_test, y_final)))
f.write("\nPrecision:"+str(metrics.precision_score(y_test, y_final)))
f.write("\nRecall:"+str(metrics.recall_score(y_test, y_final)))
f.write("\nF1-score:"+str(metrics.f1_score(y_test, y_final)))
f.write("\nMCC:"+str(metrics.matthews_corrcoef(y_test, y_final)))
f.write('\nKappa score:'+str(metrics.cohen_kappa_score(y_test,y_final)))
f.write("\nAUC:"+str(auc))

f.write("\n\n Predictions"+str(y_final))

group_names = ['True Neg','False Pos','False Neg','True Pos']

group_counts = ["{0:0.0f}".format(value) for value in
                cm.flatten()]

group_percentages = ["{0:.2%}".format(value) for value in
                     cm.flatten()/np.sum(cm)]

labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]

labels = np.asarray(labels).reshape(2,2)

ax = sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')

ax.set_title('Logistic Regression Confusion Matrix\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])
plt.savefig(path_to_file+'LG_CM.png',bbox_inches='tight')
## Display the visualization of the Confusion Matrix.
plt.show()


print("############################################################### KNN #############################################################################")
from sklearn.neighbors import KNeighborsClassifier
best=[0]
number=[]

classifier_KNN=KNeighborsClassifier(n_neighbors=13, metric="minkowski", p=1)
classifier_KNN.fit(xtrain,y_train)
y_pred_KNN=classifier_KNN.predict(xtest)
print(y_pred_KNN)
print("Predictions KNN",y_pred_KNN)


accuracy_KNN=metrics.balanced_accuracy_score(y_test, y_pred_KNN)
f1_score_KNN=metrics.f1_score(y_test, y_pred_KNN)
MCC_KNN=metrics.matthews_corrcoef(y_test, y_pred_KNN)
print(classification_report(y_test,y_pred_KNN))
print(confusion_matrix(y_test,y_pred_KNN))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_KNN))
print("Balanced Accuracy:",metrics.balanced_accuracy_score(y_test, y_pred_KNN))
print("Precision:",metrics.precision_score(y_test, y_pred_KNN))
print("Recall:",metrics.recall_score(y_test, y_pred_KNN))
print("F1-score:",metrics.f1_score(y_test, y_pred_KNN))
print("MCC:",metrics.matthews_corrcoef(y_test, y_pred_KNN))
print('Kappa score:',metrics.cohen_kappa_score(y_test,y_pred_KNN))
y_pred_proba = classifier_KNN.predict_proba(xtest)[::,1]
fpr2, tpr2, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
print("AUC:",auc)

cm=confusion_matrix(y_test,y_pred_KNN)
FPR_KNN=cm[0][1]/ (cm[0][1]+cm[0][0])
group_names = ['True Neg','False Pos','False Neg','True Pos']

group_counts = ["{0:0.0f}".format(value) for value in
                cm.flatten()]

group_percentages = ["{0:.2%}".format(value) for value in
                     cm.flatten()/np.sum(cm)]

labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]

labels = np.asarray(labels).reshape(2,2)

ax = sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')

ax.set_title('KNN Confusion Matrix\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])
plt.savefig(path_to_file+'KNN_CM.png',bbox_inches='tight')
## Display the visualization of the Confusion Matrix.
plt.show()

f.write("\n------------------------------------ KNN ----------------------------------------------------------\n")

f.write(str(confusion_matrix(y_test,y_pred_KNN)))
f.write("\nAccuracy:"+ str(metrics.accuracy_score(y_test, y_pred_KNN)))
f.write("\nBalanced Accuracy:"+str(metrics.balanced_accuracy_score(y_test, y_pred_KNN)))
f.write("\nPrecision:"+str(metrics.precision_score(y_test, y_pred_KNN)))
f.write("\nRecall:"+str(metrics.recall_score(y_test, y_pred_KNN)))
f.write("\nF1-score:"+str(metrics.f1_score(y_test, y_pred_KNN)))
f.write("\nMCC:"+str(metrics.matthews_corrcoef(y_test, y_pred_KNN)))
f.write('\nKappa score:'+str(metrics.cohen_kappa_score(y_test,y_pred_KNN)))
f.write("\nAUC:"+str(auc))

f.write("\n\n Predictions"+str(y_pred_KNN))

####################################### k cross validation results before tuning ###############################################


print('------------------------------------Grid Search Results KNN----------------------------------------')
hyperparameters=dict()
hyperparameters['metric'] =['minkowski','euclidean','manhattan']
#hyperparameters['metric'] =['minkowski']

hyperparameters['n_neighbors'] = list(range(5,50))
hyperparameters['p']=[1,2]
#hyperparameters['p']=[1,2]

grid_search_KNN = GridSearchCV(classifier_KNN, hyperparameters, refit=True, verbose=3,cv=cv,scoring="f1")
result=grid_search_KNN.fit(xtrain,y_train)

best_KNN=result.best_estimator_
print('-------------------------------GRID SEARCH RESULTS KNN--------------------------------')
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)
y_pred=best_KNN.predict(xtest)


print(confusion_matrix(y_test,y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Balanced Accuracy:",metrics.balanced_accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("F1-score:",metrics.f1_score(y_test, y_pred))
print("MCC:",metrics.matthews_corrcoef(y_test, y_pred))
print('Kappa score:',metrics.cohen_kappa_score(y_test,y_pred))
y_pred_proba = best_KNN.predict_proba(xtest)[::,1]
fpr2, tpr2, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
print("AUC:",auc)
print('----------------------------------cross validation results KNN-------------------')
cv1=StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
score= cross_validate(classifier_KNN,xtrain,y_train,cv=cv1 , scoring=_scoring,return_train_score=True)
scores=pd.DataFrame(score)
print(scores)
print("Mean of parameters {} %".format(scores.mean()*100))

train_KNN_cross_result_f1=scores['train_f1'].mean()*100
test_KNN_cross_result_f1=scores['test_f1'].mean()*100
train_KNN_cross_result_roc=scores['train_roc_auc'].mean()*100
test_KNN_cross_result_roc=scores['test_roc_auc'].mean()*100
train_KNN_cross_result_BA=scores['train_balanced_accuracy'].mean()*100
test_KNN_cross_result_BA=scores['test_balanced_accuracy'].mean()*100





############################## SVM ##############################################################


from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
classifier_SVM= SVC(kernel="rbf",random_state=0,gamma=0.8,C=0.75)
classifier_SVM.fit(xtrain,y_train)
print("############################################################ SVM ###################################################################")
y_pred=classifier_SVM.predict(xtest)

print("Predictions SVM",y_pred)

print(confusion_matrix(y_test,y_pred))

parameters = [{'C': [0.25, 0.5, 0.75, 1], 'kernel': ['linear'],'random_state':[0]},
              {'C': [0.25, 0.5, 0.75, 1], 'kernel': ['rbf',"poly"], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],'random_state':[0]},
              ]
grid = GridSearchCV(classifier_SVM, parameters,scoring="f1", verbose=3, cv=cv)
result=grid.fit(xtrain,y_train)

best_KSVM=result.best_estimator_
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)
y_pred=best_KSVM.predict(xtest)



cm=confusion_matrix(y_test,y_pred)
FPR_SVM=cm[0][1]/ (cm[0][1]+cm[0][0])
group_names = ['True Neg','False Pos','False Neg','True Pos']

group_counts = ["{0:0.0f}".format(value) for value in
                cm.flatten()]

group_percentages = ["{0:.2%}".format(value) for value in
                     cm.flatten()/np.sum(cm)]

labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]

labels = np.asarray(labels).reshape(2,2)

ax = sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')

ax.set_title('SVM Confusion Matrix\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])
plt.savefig(path_to_file+'SVM_CM.png',bbox_inches='tight')
## Display the visualization of the Confusion Matrix.
plt.show()
accuracy_KSVM=metrics.accuracy_score(y_test, y_pred)
f1_score_KSVM=metrics.f1_score(y_test, y_pred)
MCC_KSVM=metrics.matthews_corrcoef(y_test, y_pred)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Balanced Accuracy:",metrics.balanced_accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("F1-score:",metrics.f1_score(y_test, y_pred))
print("MCC:",metrics.matthews_corrcoef(y_test, y_pred))
print('Kappa score:',metrics.cohen_kappa_score(y_test,y_pred))
fpr4, tpr4, _ = metrics.roc_curve(y_test,  y_pred)
auc = metrics.roc_auc_score(y_test, y_pred)
print("AUC:",auc)

f.write("\n------------------------------------ SVM  ----------------------------------------------------------\n")
f.write(str(confusion_matrix(y_test,y_pred)))
f.write("\nAccuracy:"+ str(metrics.accuracy_score(y_test, y_pred)))
f.write("\nBalanced Accuracy:"+str(metrics.balanced_accuracy_score(y_test, y_pred)))
f.write("\nPrecision:"+str(metrics.precision_score(y_test, y_pred)))
f.write("\nRecall:"+str(metrics.recall_score(y_test, y_pred)))
f.write("\nF1-score:"+str(metrics.f1_score(y_test, y_pred)))
f.write("\nMCC:"+str(metrics.matthews_corrcoef(y_test, y_pred)))
f.write('\nKappa score:'+str(metrics.cohen_kappa_score(y_test,y_pred)))
f.write("\nAUC:"+str(auc))

f.write("+\n\n Predictions"+str(y_pred))

cv1=StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
score= cross_validate(classifier_SVM,xtrain,y_train,cv=cv1 , scoring=_scoring,return_train_score=True)
scores=pd.DataFrame(score)


train_SVM_cross_result_f1=scores['train_f1'].mean()*100
test_SVM_cross_result_f1=scores['test_f1'].mean()*100
train_SVM_cross_result_roc=scores['train_roc_auc'].mean()*100
test_SVM_cross_result_roc=scores['test_roc_auc'].mean()*100
train_SVM_cross_result_BA=scores['train_balanced_accuracy'].mean()*100
test_SVM_cross_result_BA=scores['test_balanced_accuracy'].mean()*100


##############################  DECISION TREE  ##############################################################
from sklearn.tree import DecisionTreeClassifier
classifier_DT = DecisionTreeClassifier(criterion = 'gini', random_state = 0,min_samples_split=20,
                                    min_samples_leaf=10, max_depth=20)
model=classifier_DT.fit(xtrain, y_train)


print("############################### DECISION TREE ########################################")
y_pred=classifier_DT.predict(xtest)

cm=confusion_matrix(y_test,y_pred)
FPR_DT=cm[0][1]/ (cm[0][1]+cm[0][0])
group_names = ['True Neg','False Pos','False Neg','True Pos']

group_counts = ["{0:0.0f}".format(value) for value in
                cm.flatten()]

group_percentages = ["{0:.2%}".format(value) for value in
                     cm.flatten()/np.sum(cm)]

labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]

labels = np.asarray(labels).reshape(2,2)

ax = sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')

ax.set_title('Decision Tree Confusion Matrix\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])
plt.savefig(path_to_file+'DT_CM.png',bbox_inches='tight')
## Display the visualization of the Confusion Matrix.
plt.show()


print(confusion_matrix(y_test,y_pred))
accuracy_DT=metrics.balanced_accuracy_score(y_test, y_pred)
f1_score_DT=metrics.f1_score(y_test, y_pred)
MCC_DT=metrics.matthews_corrcoef(y_test, y_pred)
cv1=StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
score= cross_validate(classifier_DT,xtrain,y_train,cv=cv1 , scoring=_scoring,return_train_score=True)
scores=pd.DataFrame(score)
train_DT_cross_result_f1=scores['train_f1'].mean()*100
test_DT_cross_result_f1=scores['test_f1'].mean()*100
train_DT_cross_result_roc=scores['train_roc_auc'].mean()*100
test_DT_cross_result_roc=scores['test_roc_auc'].mean()*100
train_DT_cross_result_BA=scores['train_balanced_accuracy'].mean()*100
test_DT_cross_result_BA=scores['test_balanced_accuracy'].mean()*100


print(scores)
print("Mean of parameters {} %".format(scores.mean()*100))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Balanced Accuracy:",metrics.balanced_accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("F1-score:",metrics.f1_score(y_test, y_pred))
print("MCC:",metrics.matthews_corrcoef(y_test, y_pred))
print('Kappa score:',metrics.cohen_kappa_score(y_test,y_pred))
fpr5, tpr5, _ = metrics.roc_curve(y_test,  y_pred)
auc = metrics.roc_auc_score(y_test, y_pred)
print("AUC:",auc)

'''print("-------------------------------------GRID SEARCH RESULTS DECISION TREE -------------------------------------------------")

# Create the parameter grid based on the results of random search
params = {
    'max_depth': [2, 3, 5, 10, 20, 25,30],
    'min_samples_leaf': [5, 10, 20, 50, 100],
    'criterion': ["gini", "entropy"],
    'random_state':[0]
}
grid_search = GridSearchCV(estimator=classifier_DT,
                           param_grid=params,
                           cv=cv, verbose=3, scoring = "f1")



result=grid_search.fit(xtrain,y_train)

best_DT=result.best_estimator_
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

y_pred=best_DT.predict(xtest)

print(confusion_matrix(y_test,y_pred))
print(scores)
print("Mean of parameters {} %".format(scores.mean()*100))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Balanced Accuracy:",metrics.balanced_accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("F1-score:",metrics.f1_score(y_test, y_pred))
print("MCC:",metrics.matthews_corrcoef(y_test, y_pred))
print('Kappa score:',metrics.cohen_kappa_score(y_test,y_pred))
fpr5, tpr5, _ = metrics.roc_curve(y_test,  y_pred)
auc = metrics.roc_auc_score(y_test, y_pred)
print("AUC:",auc)
'''
f.write("\n------------------------------------ DECISION TREE ----------------------------------------------------------\n")
f.write(str(confusion_matrix(y_test,y_pred)))
f.write("\nAccuracy:"+ str(metrics.accuracy_score(y_test, y_pred)))
f.write("\nBalanced Accuracy:"+str(metrics.balanced_accuracy_score(y_test, y_pred)))
f.write("\nPrecision:"+str(metrics.precision_score(y_test, y_pred)))
f.write("\nRecall:"+str(metrics.recall_score(y_test, y_pred)))
f.write("\nF1-score:"+str(metrics.f1_score(y_test, y_pred)))
f.write("\nMCC:"+str(metrics.matthews_corrcoef(y_test, y_pred)))
f.write('\nKappa score:'+str(metrics.cohen_kappa_score(y_test,y_pred)))
f.write("\nAUC:"+str(auc))
f.write("\n\n Predictions"+str(y_pred))
from sklearn import tree
text_representation = tree.export_text(classifier_DT)
print(text_representation)


'''fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (5,5), dpi=300)
_ = tree.plot_tree(model,feature_names=X_train.columns.values,filled=True)
plt.show()
'''




'''sort = classifier.feature_importances_.argsort()
plt.barh(df1.columns[sort], classifier.feature_importances_[sort])
plt.xlabel("Feature Importance")
plt.show()'''


############################################# RANDOM FOREST ################################################

from sklearn.ensemble import RandomForestClassifier

from sklearn import decomposition, datasets
from sklearn import tree
from sklearn.pipeline import Pipeline


classifier_RF = RandomForestClassifier(n_estimators =1000, criterion = 'entropy', random_state = 0, max_depth=30, min_samples_leaf=1,min_samples_split=2)
classifier_RF.fit(xtrain, y_train)
print("############################### RANDOM FOREST ########################################")
y_pred=classifier_RF.predict(xtest)
print(confusion_matrix(y_test,y_pred))

cm=confusion_matrix(y_test,y_pred)
FPR_RF=cm[0][1]/ (cm[0][1]+cm[0][0])
group_names = ['True Neg','False Pos','False Neg','True Pos']

group_counts = ["{0:0.0f}".format(value) for value in
                cm.flatten()]

group_percentages = ["{0:.2%}".format(value) for value in
                     cm.flatten()/np.sum(cm)]

labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]

labels = np.asarray(labels).reshape(2,2)

ax = sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')

ax.set_title('Random Forest Confusion Matrix\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])
plt.savefig(path_to_file+'RF_CM.png',bbox_inches='tight')
## Display the visualization of the Confusion Matrix.
plt.show()
accuracy_RF=metrics.balanced_accuracy_score(y_test, y_pred)
f1_score_RF=metrics.f1_score(y_test, y_pred)
MCC_RF=metrics.matthews_corrcoef(y_test, y_pred)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Balanced Accuracy:",metrics.balanced_accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("F1-score:",metrics.f1_score(y_test, y_pred))
print("MCC:",metrics.matthews_corrcoef(y_test, y_pred))
print('Kappa score:',metrics.cohen_kappa_score(y_test,y_pred))
fpr6, tpr6, _ = metrics.roc_curve(y_test,  y_pred)
auc = metrics.roc_auc_score(y_test, y_pred)
print("AUC:",auc)



################################################ GRID SEARH RANDOM FOREST ##################################################



'''print("-------------------------------------GRID SEARCH RESULTS RANDOM FOREST -------------------------------------------------")

# Create the parameter grid based on the results of random search
params = {

'criterion': ["gini", "entropy"],
'max_depth': [10, 20, 30],
'min_samples_leaf': [1, 2],
'min_samples_split': [2, 5],
'n_estimators': [10,200, 400, 600, 800, 1000],
'random_state':[0]
}

grid_search = GridSearchCV(estimator=classifier_RF,
                           param_grid=params,
                           cv=cv, verbose=3, scoring = "f1")



result=grid_search.fit(xtrain,y_train)

best_RF=result.best_estimator_
print('Best Score: %s' % result.best_score_)
print("Best Hyperparameters: %s" % result.best_params_)

y_pred=best_RF.predict(xtest)'''


print(confusion_matrix(y_test,y_pred))
f.write("\n------------------------------------ RANDOM FOREST ----------------------------------------------------------\n")
f.write(str(confusion_matrix(y_test,y_pred)))
f.write("\nAccuracy:"+ str(metrics.accuracy_score(y_test, y_pred)))
f.write("\nBalanced Accuracy:"+str(metrics.balanced_accuracy_score(y_test, y_pred)))
f.write("\nPrecision:"+str(metrics.precision_score(y_test, y_pred)))
f.write("\nRecall:"+str(metrics.recall_score(y_test, y_pred)))
f.write("\nF1-score:"+str(metrics.f1_score(y_test, y_pred)))
f.write("\nMCC:"+str(metrics.matthews_corrcoef(y_test, y_pred)))
f.write('\nKappa score:'+str(metrics.cohen_kappa_score(y_test,y_pred)))
f.write("\nAUC:"+str(auc))

f.write("\n\n Predictions"+str(y_pred))
'''importances = classifier_RF.feature_importances_
std = np.std([tree.feature_importances_ for tree in classifier_RF.estimators_], axis=0)
forest_importances = pd.Series(importances, index=df1.columns[1:len(df1.columns)])'''
cv1=StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
score= cross_validate(classifier_RF,xtrain,y_train,cv=cv1 , scoring=_scoring,return_train_score=True)
scores=pd.DataFrame(score)

train_RF_cross_result_f1=scores['train_f1'].mean()*100
test_RF_cross_result_f1=scores['test_f1'].mean()*100
train_RF_cross_result_roc=scores['train_roc_auc'].mean()*100
test_RF_cross_result_roc=scores['test_roc_auc'].mean()*100
train_RF_cross_result_BA=scores['train_balanced_accuracy'].mean()*100
test_RF_cross_result_BA=scores['test_balanced_accuracy'].mean()*100


print(scores)
print("Mean of parameters {} %".format(scores.mean()*100))
print("Mean of parameters {} %".format(scores['test_f1'].mean()*100))
'''
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()'''
#################################################################### XG BOOST ###################################
print("------------------------------------XGBOOST------------------------------------------------")
# Training XGBoost on the Training set

classifier_XG = XGBClassifier(colsample_bytree=0.6, gamma=1.5,max_depth=4,min_child_weight=1,subsample=1.0,random_state=0)
classifier_XG.fit(xtrain, y_train)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier_XG.predict(xtest)
cm = confusion_matrix(y_test, y_pred)
print(cm)
group_names = ['True Neg','False Pos','False Neg','True Pos']

group_counts = ["{0:0.0f}".format(value) for value in
                cm.flatten()]

group_percentages = ["{0:.2%}".format(value) for value in
                     cm.flatten()/np.sum(cm)]

labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]

labels = np.asarray(labels).reshape(2,2)

ax = sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')

ax.set_title('XG BOOST Confusion Matrix\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])
plt.savefig(path_to_file+'XG_CM.png',bbox_inches='tight')




FPR_XG=cm[0][1]/ (cm[0][1]+cm[0][0])
accuracy_score(y_test, y_pred)



print("Mean of parameters {} %".format(scores.mean()*100))
MCC_XG=metrics.matthews_corrcoef(y_test, y_pred)
f1_score_XG=metrics.f1_score(y_test, y_pred)
accuracy_XG=metrics.balanced_accuracy_score(y_test, y_pred)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Balanced Accuracy:",metrics.balanced_accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("F1-score:",metrics.f1_score(y_test, y_pred))
print("MCC:",metrics.matthews_corrcoef(y_test, y_pred))
print('Kappa score:',metrics.cohen_kappa_score(y_test,y_pred))
fpr7, tpr7, _ = metrics.roc_curve(y_test,  y_pred)
auc = metrics.roc_auc_score(y_test, y_pred)
print("AUC:",auc)
'''
params = {
        'min_child_weight': [1, 5],
        'gamma': [0.5, 1, 1.5, 2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5],
        'random_state':[0]
        }

grid_search = GridSearchCV(estimator=classifier_XG,
                           param_grid=params,
                           cv=cv, verbose=3, scoring = "f1")



result=grid_search.fit(xtrain,y_train)

best_XG=result.best_estimator_
print('Best Score: %s' % result.best_score_)
print("Best Hyperparameters: %s" % result.best_params_)

y_pred=best_XG.predict(xtest)
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("Mean of parameters {} %".format(scores.mean()*100))

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Balanced Accuracy:",metrics.balanced_accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("F1-score:",metrics.f1_score(y_test, y_pred))
print("MCC:",metrics.matthews_corrcoef(y_test, y_pred))
print('Kappa score:',metrics.cohen_kappa_score(y_test,y_pred))
fpr8, tpr8, _ = metrics.roc_curve(y_test,  y_pred)
auc = metrics.roc_auc_score(y_test, y_pred)
print("AUC:",auc)
'''

####################################################### WRITE TO FILE ######################################################
f.write("---------------------------------------- No DoS attack - evaluation file  ----------------------------------------------------------\n")
f.write("\n------------------------------------ XG BOOST ----------------------------------------------------------\n")
f.write(str(confusion_matrix(y_test,y_pred)))
f.write("\nAccuracy:"+ str(metrics.accuracy_score(y_test, y_pred)))
f.write("\nBalanced Accuracy:"+str(metrics.balanced_accuracy_score(y_test, y_pred)))
f.write("\nPrecision:"+str(metrics.precision_score(y_test, y_pred)))
f.write("\nRecall:"+str(metrics.recall_score(y_test, y_pred)))
f.write("\nF1-score:"+str(metrics.f1_score(y_test, y_pred)))
f.write("\nMCC:"+str(metrics.matthews_corrcoef(y_test, y_pred)))
f.write('\nKappa score:'+str(metrics.cohen_kappa_score(y_test,y_pred)))
f.write("\nAUC:"+str(auc))

f.write("\n\n Predictions"+str(y_pred))
f.close()
################################################################################################################################################
cv1=StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
score= cross_validate(classifier_XG,xtrain,y_train,cv=cv1 , scoring=_scoring,return_train_score=True)
scores=pd.DataFrame(score)
print(scores)


train_XG_cross_result_f1=scores['train_f1'].mean()*100
test_XG_cross_result_f1=scores['test_f1'].mean()*100
train_XG_cross_result_roc=scores['train_roc_auc'].mean()*100
test_XG_cross_result_roc=scores['test_roc_auc'].mean()*100
train_XG_cross_result_BA=scores['train_balanced_accuracy'].mean()*100
test_XG_cross_result_BA=scores['test_balanced_accuracy'].mean()*100



########################################################## MAKE GRAPHS #########################################################
N = 6
ind = np.arange(N)  # the x locations for the groups
width = 0.27       # the width of the bars

fig = plt.figure()
ax = fig.add_subplot(111)

yvals = [  f1_score_KNN,f1_score_LR,f1_score_KSVM, f1_score_DT,f1_score_RF,f1_score_XG]
rects1 = ax.bar(ind, yvals, width,  color=(0.2, 0.4, 0.6, 0.6))
zvals = [accuracy_KNN,accuracy_LR, accuracy_KSVM,accuracy_DT,accuracy_RF,accuracy_XG]
rects2 = ax.bar(ind+width, zvals, width, color='firebrick')



ax.set_xticks(ind+width)
ax.set_title("Results from truly unseen data")
ax.set_xticklabels( ( 'KNN', 'Logistic Regression', 'SVM','Decision Tree','Random Forest', 'XGBOOST') )
ax.legend( (rects1[0], rects2[0] ), ('F1_score', 'Balanced_accuracy'),loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5  )

def autolabel(rects):
    for rect in rects:

        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1*h, '%.2f'%(h),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.savefig(path_to_file+"unseen_data.png")
plt.show()


################################################################################################################################
def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i,y[i],y[i])
list=['KNN', 'Logistic Regression', 'SVM','Decision Tree','Random Forest',"XGBOOST"]

kvals = [MCC_KNN,MCC_LR,MCC_KSVM,MCC_DT,MCC_RF,MCC_XG]


y_pos = np.arange(len(list))
threshold=0
# Basic plot
plt.bar(y_pos, kvals, color=(0.2, 0.4, 0.6, 0.6),label='MCC',width=0.3)
plt.plot([-0.5, len(list)], [threshold, threshold], "k--")
addlabels(y_pos, kvals)
plt.ylim(-1,1,1.0)
plt.title("MCC from truly unseen data")
# use the plt.xticks function to custom labels
plt.xticks(y_pos, list)
plt.legend()

plt.savefig(path_to_file+"MCC.png")
plt.show()


def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i,y[i],y[i])

list=['KNN', 'Logistic Regression', 'SVM','Decision Tree','Random Forest',"XGBOOST"]

kvals = [FPR_KNN*100,FPR_LG*100,FPR_SVM*100,FPR_DT*100,FPR_RF*100,FPR_XG*100]


y_pos = np.arange(len(list))
threshold=0
# Basic plot
plt.bar(y_pos, kvals, color=(0.2, 0.4, 0.6, 0.6),width=0.25)
#plt.plot([0, len(list)], [threshold, threshold], "k--")
#plt.ylim(-1,1,1.0)
addlabels(y_pos, kvals)
plt.ylabel("FPR %")
plt.title("False positive rate (FPR)")
# use the plt.xticks function to custom labels
plt.xticks(y_pos, list)
#plt.legend()
plt.savefig(path_to_file+"FPR_score.png")
plt.show()









################################################# k cross validation results #####################################################
N = 6
ind = np.arange(N)  # the x locations for the groups
width = 0.15       # the width of the bars

fig = plt.figure()
ax = fig.add_subplot(111)

yvals = [train_KNN_cross_result_f1,train_LG_cross_result_f1,train_SVM_cross_result_f1,train_DT_cross_result_f1,train_RF_cross_result_f1,train_XG_cross_result_f1]
rects1 = ax.bar(ind, yvals, width,   color=(0.2, 0.4, 0.6, 0.6))
zvals = [test_KNN_cross_result_f1,test_LG_cross_result_f1,test_SVM_cross_result_f1,test_DT_cross_result_f1,test_RF_cross_result_f1,test_XG_cross_result_f1]
rects2 = ax.bar(ind+width, zvals, width,color='firebrick')


kvals = [train_KNN_cross_result_BA,train_LG_cross_result_BA,train_SVM_cross_result_BA,train_DT_cross_result_BA,train_RF_cross_result_BA,train_XG_cross_result_BA]
rects3= ax.bar(ind+width*2, kvals, width, color='yellowgreen')
mvals = [test_KNN_cross_result_BA,test_LG_cross_result_BA,test_SVM_cross_result_BA,test_DT_cross_result_BA,test_RF_cross_result_BA,test_XG_cross_result_BA]
rects4 = ax.bar(ind+width+width*2, kvals, width, color='mediumpurple')

ax.set_xticks(ind+width)
ax.set_xlabel('%')
ax.set_title("K-cross validation results")
ax.set_xticklabels( ( 'KNN', 'Logistic Regression', 'SVM','Decision Tree','Random Forest','XGBOOST') )
ax.legend( (rects1[0], rects2[0], rects3[0],rects4[0]), ('Train F1_score','Test F1_score','Train BA', 'Test BA'),loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5 )

def autolabel(rects):
    for rect in rects:

        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1*h, '%.1f'%(h),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
plt.savefig(path_to_file+"cross_val_results.png")
plt.show()



