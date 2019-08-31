# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 20:51:55 2018

@author: Md. Kamrul Hasan
"""

print(__doc__)
import time
startTime = time.time()
from sklearn import svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tflearn.data_utils import to_categorical
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV

#%% New data----------------------------------------------------------
TrainNegativeFeatures_4096= pd.read_csv('AUG_Samples_2250_TrainNegFatures_4096.csv',header=None)
TrainNegativeFeatures_4096=np.array(TrainNegativeFeatures_4096)
TrainNegativeFeatures_4096=TrainNegativeFeatures_4096[1:2251,1:4097]

TrainPositiveFeatures_4096= pd.read_csv('AUG_Samples_2250_TrainPos_Faetures_4096.csv',header=None)
TrainPositiveFeatures_4096=np.array(TrainPositiveFeatures_4096)
TrainPositiveFeatures_4096=TrainPositiveFeatures_4096[1:2251,1:4097]

TestNegativeFeatures_4096= pd.read_csv('AUG_Samples_750_TestNeg_Features_4096.csv',header=None)
TestNegativeFeatures_4096=np.array(TestNegativeFeatures_4096)
TestNegativeFeatures_4096=TestNegativeFeatures_4096[1:751,1:4097]

TestPositiveFeatures_4096= pd.read_csv('AUG_Samples_750_TestPos_Features_4096.csv',header=None)
TestPositiveFeatures_4096=np.array(TestPositiveFeatures_4096)
TestPositiveFeatures_4096=TestPositiveFeatures_4096[1:751,1:4097]

DataTr=np.concatenate((TrainPositiveFeatures_4096, TrainNegativeFeatures_4096), axis=0)

DataTe=np.concatenate((TestPositiveFeatures_4096, TestNegativeFeatures_4096), axis=0)
#%% Labels-------------------------------------------------------------------
Zero_Train_Label = np.empty(2250)
Zero_Train_Label.fill(0)
One_Train_Label = np.empty(2250)
One_Train_Label.fill(1)

Zero_Test_Label = np.empty(750)
Zero_Test_Label.fill(0)
One_Test_Label = np.empty(750)
One_Test_Label.fill(1)

LabelTrain=np.concatenate((Zero_Train_Label, One_Train_Label), axis=0)
LabelTest=np.concatenate((Zero_Test_Label, One_Test_Label), axis=0)
#=====Positive and Negative Split of the Label=====
LabelPositive=np.concatenate((One_Train_Label, One_Test_Label), axis=0)
LabelNegative=np.concatenate((Zero_Train_Label, Zero_Test_Label), axis=0)

PositiveData=np.concatenate((TrainPositiveFeatures_4096, TestPositiveFeatures_4096), axis=0)
NegativeData=np.concatenate((TrainNegativeFeatures_4096,TestNegativeFeatures_4096), axis=0)

#%% Now Features Reduction
Data=np.concatenate((PositiveData,NegativeData), axis=0)
Label=np.concatenate((LabelPositive,LabelNegative), axis=0)
model = ExtraTreesClassifier()
model.fit(Data, Label)
rate=model.feature_importances_
ind=[i for i, e in enumerate(rate) if e != 0]
PositiveData=PositiveData[:,ind]
NegativeData=NegativeData[:,ind]

#%%
PositiveDataCase_1=PositiveData[0:600,:]
PositiveDataCase_2=PositiveData[600:1200,:]
PositiveDataCase_3=PositiveData[1200:1800,:]
PositiveDataCase_4=PositiveData[1800:2400,:]
PositiveDataCase_5=PositiveData[2400:3000,:]

# for the case-1
TestPosCase_1=PositiveDataCase_1
TrainValCase_1=np.concatenate((PositiveDataCase_2, PositiveDataCase_3,PositiveDataCase_4,PositiveDataCase_5), axis=0)
# for the case-2
TestPosCase_2=PositiveDataCase_2
TrainValCase_2=np.concatenate((PositiveDataCase_1, PositiveDataCase_3,PositiveDataCase_4,PositiveDataCase_5), axis=0)
# for the case-3
TestPosCase_3=PositiveDataCase_3
TrainValCase_3=np.concatenate((PositiveDataCase_1, PositiveDataCase_2,PositiveDataCase_4,PositiveDataCase_5), axis=0)
# for the case-4
TestPosCase_4=PositiveDataCase_4
TrainValCase_4=np.concatenate((PositiveDataCase_1, PositiveDataCase_2,PositiveDataCase_3,PositiveDataCase_5), axis=0)
# for the case-5
TestPosCase_5=PositiveDataCase_5
TrainValCase_5=np.concatenate((PositiveDataCase_1, PositiveDataCase_2,PositiveDataCase_3,PositiveDataCase_4), axis=0)

NegativeDataCase_1=NegativeData[0:600,:]
NegativeDataCase_2=NegativeData[600:1200,:]
NegativeDataCase_3=NegativeData[1200:1800,:]
NegativeDataCase_4=NegativeData[1800:2400,:]
NegativeDataCase_5=NegativeData[2400:3000,:]

# for the case-1
TestNegCase_1=NegativeDataCase_1
TrainValNegCase_1=np.concatenate((NegativeDataCase_2, NegativeDataCase_3,NegativeDataCase_4,NegativeDataCase_5), axis=0)

# for the case-2
TestNegCase_2=NegativeDataCase_2
TrainValNegCase_2=np.concatenate((NegativeDataCase_1, NegativeDataCase_3,NegativeDataCase_4,NegativeDataCase_5), axis=0)
# for the case-3
TestNegCase_3=NegativeDataCase_3
TrainValNegCase_3=np.concatenate((NegativeDataCase_1, NegativeDataCase_2,NegativeDataCase_4,NegativeDataCase_5), axis=0)
# for the case-4
TestNegCase_4=NegativeDataCase_4
TrainValNegCase_4=np.concatenate((NegativeDataCase_1, NegativeDataCase_2,NegativeDataCase_3,NegativeDataCase_5), axis=0)
# for the case-5
TestNegCase_5=NegativeDataCase_5
TrainValNegCase_5=np.concatenate((NegativeDataCase_1, NegativeDataCase_2,NegativeDataCase_3,NegativeDataCase_4), axis=0)

#==Concatenate Negative 1-4000 and Positive 4001-8000==
Data=np.concatenate((NegativeData, PositiveData), axis=0)
#for case-1
TrainValCase_1=np.concatenate((TrainValNegCase_1, TrainValCase_1), axis=0)
TestCase_1=np.concatenate((TestNegCase_1, TestPosCase_1), axis=0)
#for case-2
TrainValCase_2=np.concatenate((TrainValNegCase_2, TrainValCase_2), axis=0)
TestCase_2=np.concatenate((TestNegCase_2, TestPosCase_2), axis=0)
#for case-3
TrainValCase_3=np.concatenate((TrainValNegCase_3, TrainValCase_3), axis=0)
TestCase_3=np.concatenate((TestNegCase_3, TestPosCase_3), axis=0)
#for case-4
TrainValCase_4=np.concatenate((TrainValNegCase_4, TrainValCase_4), axis=0)
TestCase_4=np.concatenate((TestNegCase_4, TestPosCase_4), axis=0)
#for case-5
TrainValCase_5=np.concatenate((TrainValNegCase_5, TrainValCase_5), axis=0)
TestCase_5=np.concatenate((TestNegCase_5, TestPosCase_5), axis=0)

PositiveLabelCase_1=LabelPositive[0:600]
PositiveLabelCase_2=LabelPositive[600:1200]
PositiveLabelCase_3=LabelPositive[1200:1800]
PositiveLabelCase_4=LabelPositive[1800:2400]
PositiveLabelCase_5=LabelPositive[2400:3000]
# for the case-1
LabelTestPosCase_1=PositiveLabelCase_1
LabelTrainValPosCase_1=np.concatenate((PositiveLabelCase_2, PositiveLabelCase_3, PositiveLabelCase_4, PositiveLabelCase_5), axis=0)
# for the case-2
LabelTestPosCase_2=PositiveLabelCase_2
LabelTrainValPosCase_2=np.concatenate((PositiveLabelCase_1, PositiveLabelCase_3, PositiveLabelCase_4, PositiveLabelCase_5), axis=0)
# for the case-3
LabelTestPosCase_3=PositiveLabelCase_3
LabelTrainValPosCase_3=np.concatenate((PositiveLabelCase_1, PositiveLabelCase_2, PositiveLabelCase_4, PositiveLabelCase_5), axis=0)
# for the case-4
LabelTestPosCase_4=PositiveLabelCase_4
LabelTrainValPosCase_4=np.concatenate((PositiveLabelCase_1, PositiveLabelCase_2, PositiveLabelCase_3, PositiveLabelCase_5), axis=0)
# for the case-5
LabelTestPosCase_5=PositiveLabelCase_5
LabelTrainValPosCase_5=np.concatenate((PositiveLabelCase_1, PositiveLabelCase_2, PositiveLabelCase_3, PositiveLabelCase_4), axis=0)

NegativeLabelCase_1=LabelNegative[0:600]
NegativeLabelCase_2=LabelNegative[600:1200]
NegativeLabelCase_3=LabelNegative[1200:1800]
NegativeLabelCase_4=LabelNegative[1800:2400]
NegativeLabelCase_5=LabelNegative[2400:3000]

# for the case-1
LabelTestNegCase_1=NegativeLabelCase_1
LabelTrainValNegCase_1=np.concatenate((NegativeLabelCase_2, NegativeLabelCase_3, NegativeLabelCase_4, NegativeLabelCase_5), axis=0)
# for the case-2
LabelTestNegCase_2=NegativeLabelCase_2
LabelTrainValNegCase_2=np.concatenate((NegativeLabelCase_1, NegativeLabelCase_3, NegativeLabelCase_4, NegativeLabelCase_5), axis=0)
# for the case-3
LabelTestNegCase_3=NegativeLabelCase_3
LabelTrainValNegCase_3=np.concatenate((NegativeLabelCase_1, NegativeLabelCase_2, NegativeLabelCase_4, NegativeLabelCase_5), axis=0)
# for the case-3
LabelTestNegCase_3=NegativeLabelCase_3
LabelTrainValNegCase_3=np.concatenate((NegativeLabelCase_1, NegativeLabelCase_2, NegativeLabelCase_4, NegativeLabelCase_5), axis=0)
# for the case-4
LabelTestNegCase_4=NegativeLabelCase_4
LabelTrainValNegCase_4=np.concatenate((NegativeLabelCase_1, NegativeLabelCase_2, NegativeLabelCase_3, NegativeLabelCase_5), axis=0)
# for the case-5
LabelTestNegCase_5=NegativeLabelCase_5
LabelTrainValNegCase_5=np.concatenate((NegativeLabelCase_1, NegativeLabelCase_2, NegativeLabelCase_3, NegativeLabelCase_4), axis=0)

#====Label concatenate. lebel 0 first then Label 1.====
NewLabel=np.concatenate((LabelNegative, LabelPositive), axis=0)
#for Case -1
TrainValLabelcase_1=np.concatenate((LabelTrainValNegCase_1, LabelTrainValPosCase_1), axis=0)
TestLabelcase_1=np.concatenate((LabelTestNegCase_1, LabelTestPosCase_1), axis=0)
#for Case -2
TrainValLabelcase_2=np.concatenate((LabelTrainValNegCase_2, LabelTrainValPosCase_2), axis=0)
TestLabelcase_2=np.concatenate((LabelTestNegCase_2, LabelTestPosCase_2), axis=0)
#for Case -3
TrainValLabelcase_3=np.concatenate((LabelTrainValNegCase_3, LabelTrainValPosCase_3), axis=0)
TestLabelcase_3=np.concatenate((LabelTestNegCase_3, LabelTestPosCase_3), axis=0)
#for Case -4
TrainValLabelcase_4=np.concatenate((LabelTrainValNegCase_4, LabelTrainValPosCase_4), axis=0)
TestLabelcase_4=np.concatenate((LabelTestNegCase_4, LabelTestPosCase_4), axis=0)
#for Case -5
TrainValLabelcase_5=np.concatenate((LabelTrainValNegCase_5, LabelTrainValPosCase_5), axis=0)
TestLabelcase_5=np.concatenate((LabelTestNegCase_5, LabelTestPosCase_5), axis=0)
#%% This sections is for SVM
Kernel='rbf' # here you can select the kernel.

#%% This is for nu-SVM
tuned_parameters = [{'kernel': [Kernel],'nu':[0.001,.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
#tuned_parameters = [{'kernel': [Kernel],'nu':[0.01]}]

classifier = svm.NuSVC()

TrainValidationList=[TrainValCase_1,TrainValCase_2, TrainValCase_3, TrainValCase_4, TrainValCase_5]

TrainValidationListLAbel=[TrainValLabelcase_1,TrainValLabelcase_2, TrainValLabelcase_3, TrainValLabelcase_4, TrainValLabelcase_5]

OptimumNU=[]
AUCvar=[]
for ind in range(len(TrainValidationList)):
    clf = GridSearchCV(classifier, tuned_parameters, cv=5,scoring='roc_auc',n_jobs=1)
    data=TrainValidationList[ind]
    dataLabel=TrainValidationListLAbel[ind]
    clf.fit(data, dataLabel)
    AUCvar.append(clf.cv_results_['mean_test_score'])
    OptimumNU.append(clf.best_params_)


Train=[]
TrainLabel=[]
Test=[TestCase_1,TestCase_2,TestCase_3,TestCase_4,TestCase_5]
TestLabel=[TestLabelcase_1,TestLabelcase_2,TestLabelcase_3,TestLabelcase_4,TestLabelcase_5]

for ind in range(len(TrainValidationList)):
    data=TrainValidationList[ind]
    dataLabel=TrainValidationListLAbel[ind]
    X_train, X_test, y_train, y_test = train_test_split(
    data, dataLabel, test_size=0.5, random_state=42)
    Train.append(X_train)
    TrainLabel.append(y_train)

maxAUC=[]
for ind in range(len(AUCvar)):
    maxAUC.append(np.amax(AUCvar[ind]))

maxx=maxAUC.index(max(maxAUC))
nucase=OptimumNU[maxx]
nucase_1=nucase['nu']

AROC=[]
plotIndex=1
classifier=svm.NuSVC(nu=nucase_1, kernel=Kernel,probability=True)
for ind in range(5):
    probas_ = classifier.fit(Train[ind], TrainLabel[ind]).predict_proba(Test[ind])
    LabelTest_1Hot= to_categorical(TestLabel[ind],2)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(LabelTest_1Hot[:, i], probas_[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    AUCC=(roc_auc[0]+roc_auc[1])/2
    print()
    print("Area Under ROC (AUC) for SVM: {}".format(AUCC))
    AROC.append(AUCC)
    lw = 2
    plt.grid(True)
    plt.plot(fpr[1], tpr[1],lw=2,label='ROC for Case- %d ' % (plotIndex))
    plotIndex += 1
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver operating characteristic (ROC) for nu-SVM')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',label='Bad Classification')
plt.legend(loc="lower right")
plt.show()
mean=np.mean(AROC)
std=np.std(AROC)
print("%0.3f (+/-%0.03f)" % (mean, std))

#%% This sections is for CSVM
tuned_parameters = [{'kernel': [Kernel],'C':[0.0001,0.001,0.01,0.1,1,10,100,1000,10000]}]
#tuned_parameters = [{'kernel': [Kernel],'C':[0.01]}]

classifier = svm.SVC()

TrainValidationList=[TrainValCase_1,TrainValCase_2, TrainValCase_3, TrainValCase_4, TrainValCase_5]

TrainValidationListLAbel=[TrainValLabelcase_1,TrainValLabelcase_2, TrainValLabelcase_3, TrainValLabelcase_4, TrainValLabelcase_5]

OptimumC=[]
AUCvarC=[]
for ind in range(len(TrainValidationList)):
    clf = GridSearchCV(classifier, tuned_parameters, cv=5,scoring='roc_auc',n_jobs=1)
    data=TrainValidationList[ind]
    dataLabel=TrainValidationListLAbel[ind]
    clf.fit(data, dataLabel)
    AUCvarC.append(clf.cv_results_['mean_test_score'])
    OptimumC.append(clf.best_params_)


Train=[]
TrainLabel=[]
Test=[TestCase_1,TestCase_2,TestCase_3,TestCase_4,TestCase_5]
TestLabel=[TestLabelcase_1,TestLabelcase_2,TestLabelcase_3,TestLabelcase_4,TestLabelcase_5]

for ind in range(len(TrainValidationList)):
    data=TrainValidationList[ind]
    dataLabel=TrainValidationListLAbel[ind]
    X_train, X_test, y_train, y_test = train_test_split(
    data, dataLabel, test_size=0.5, random_state=42)
    Train.append(X_train)
    TrainLabel.append(y_train)

maxAUCC=[]
for ind in range(len(AUCvarC)):
    maxAUCC.append(np.amax(AUCvarC[ind]))

maxx=maxAUCC.index(max(maxAUCC))
C_case=OptimumC[maxx]
C_case_1=C_case['C']

AROC=[]
classifier=svm.SVC(C=C_case_1, kernel=Kernel,probability=True)
plotIndex=1
for ind in range(5):
    probas_ = classifier.fit(Train[ind], TrainLabel[ind]).predict_proba(Test[ind])
    LabelTest_1Hot= to_categorical(TestLabel[ind],2)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(LabelTest_1Hot[:, i], probas_[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    AUCC=(roc_auc[0]+roc_auc[1])/2
    print()
    print("Area Under ROC (AUC) for SVM: {}".format(AUCC))
    AROC.append(AUCC)
    plt.grid(True)
    plt.plot(fpr[1], tpr[1],lw=2,label='ROC for Case- %d ' % (plotIndex))
    plotIndex += 1
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver operating characteristic (ROC) for C-SVM')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',label='Bad Classification')
plt.legend(loc="lower right")
plt.show()
mean=np.mean(AROC)
std=np.std(AROC)
print("%0.3f (+/-%0.03f)" % (mean, std))

#%%
endTime = time.time()
print('It took {0:0.1f} seconds'.format(endTime - startTime))