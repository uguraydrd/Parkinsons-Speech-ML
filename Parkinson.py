# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 20:53:57 2019

@author: uğuray
"""

#General Libraries
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#################################################################################################
#Let's read the data
df = pd.read_csv("parkinson.csv")
df2 = pd.read_excel("persons.xlsx") #Created ourselves
df=pd.concat([df,df2],1)
del df2

##########################################################
#Know the data
df.head() 
df.tail() 
df.shape 
df.info() 
describe = df.describe().T

##########################################################
#Number of each class
print("Patient:", df[df['Class'] == 1].shape[0])
print("Healthy:", df[df['Class'] == 0].shape[0])

#################################################################################################
#VISUALIZATION

#Visualizations of variables and box plots viewed
#Outlier values and variables was dropped

#Correlation Matrix
plt.title('Correlation Matrix', y=1.05, size=16)
corr = df.drop(["Class"], 1).corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(corr, annot=True, fmt=".2f", mask=mask,  
                     linewidths=.5, cbar_kws={"shrink": .5})
plt.xticks(rotation=270) 

#As we see in the correlation matrix, drop the five variables
df.drop(["MDVP:PPQ", 
         "Jitter:DDP", 
         "MDVP:Shimmer", 
         "MDVP:Shimmer(dB)", 
         "Shimmer:APQ3"], 1, inplace=True)
    
########################################################## 
index_class_h=[]
for i in range(144):
    index_class_h.append(i)
    
index_class_s=[]
for i in range(48):
    index_class_s.append(i)
    
##########################################################
#MDVP:Flo(Hz) 
plt.scatter(x=index_class_h, y=df[df['Class']==1]["MDVP:Flo(Hz)"][:144].sort_values(), 
            color='r', label="Patient")
plt.scatter(x=index_class_s, y=df[df['Class']==0]["MDVP:Flo(Hz)"][:48].sort_values(), 
            color='b', label="Healthy")
plt.title('Minimum Audio Frequency', fontweight='bold', y=1.01, fontsize=12)
plt.legend(loc='upper left', bbox_to_anchor=(0.27, -0.06), ncol=2, prop={'size': 10})
plt.yticks(np.arange(60, 260, 10))
plt.xticks(np.arange(0, 150, 20))
plt.ylabel('MDVP:Flo(Hz)', fontsize='large', fontweight='bold')
plt.grid(lw = 0.35)

sns.set(style="whitegrid")
sns.boxplot(x='Persons', y='MDVP:Flo(Hz)', data=df, palette="colorblind", hue='Class', 
            linewidth=1.5, dodge=False, order=["Healthy", "Patient"]
            ).set_title("Minimum Audio Frequency", fontweight='bold')

##########################################################
#MDVP:Fo(Hz)
plt.scatter(x=index_class_h, y=df[df['Class']==1]["MDVP:Fo(Hz)"][:144].sort_values(), 
            color='r', label="Patient")
plt.scatter(x=index_class_s, y=df[df['Class']==0]["MDVP:Fo(Hz)"][:48].sort_values(), 
            color='b', label="Healthy")
plt.title('Average Audio Frequency', fontweight='bold', y=1.01, fontsize=12)
plt.legend(loc='upper left', bbox_to_anchor=(0.27, -0.06), ncol=2, prop={'size': 10})
plt.yticks(np.arange(80, 280, 10))
plt.xticks(np.arange(0, 150, 20))
plt.ylabel('MDVP:Fo(Hz)', fontsize='large', fontweight='bold')
plt.grid(lw = 0.35)

sns.set(style="whitegrid")
sns.boxplot(x='Persons', y='MDVP:Fo(Hz)', data=df, palette="colorblind", hue='Class', 
            linewidth=1.5, dodge=False, order=["Healthy", "Patient"]
            ).set_title("Average Audio Frequency", fontweight='bold')

##########################################################
#MDVP:Fhi(Hz)
plt.scatter(x=index_class_h, y=df[df['Class']==1]["MDVP:Fhi(Hz)"][:144].sort_values(), 
            color='r', label="Patient")
plt.scatter(x=index_class_s, y=df[df['Class']==0]["MDVP:Fhi(Hz)"][:48].sort_values(), 
            color='b', label="Healthy")
plt.title('Maximum Audio Frequency', fontweight='bold', y=1.01, fontsize=12)
plt.legend(loc='upper left', bbox_to_anchor=(0.27, -0.06), ncol=2, prop={'size': 10})
plt.yticks(np.arange(90, 650, 30))
plt.xticks(np.arange(0, 150, 20))
plt.ylabel('MDVP:Fhi(Hz)', fontsize='large', fontweight='bold')
plt.grid(lw = 0.35)

sns.set(style="whitegrid")
sns.boxplot(x='Persons', y='MDVP:Fhi(Hz)', data=df, palette="colorblind", hue='Class', 
            linewidth=1.5, dodge=False, order=["Healthy", "Patient"]
            ).set_title("Maximum Audio Frequency", fontweight='bold')
df.drop([183,184,73,101,147], inplace=True)

##########################################################
#MDVP:Jitter(%)
plt.scatter(x=index_class_h, y=df[df['Class']==1]["MDVP:Jitter(%)"][:141].sort_values(), 
            color='r', label="Patient")
plt.scatter(x=index_class_s, y=df[df['Class']==0]["MDVP:Jitter(%)"][:46].sort_values(), 
            color='b', label="Healthy")
plt.title('Frequency Percentage Irregularity Ratio', fontweight='bold', y=1.01, fontsize=12)
plt.legend(loc='upper left', bbox_to_anchor=(0.27, -0.06), ncol=2, prop={'size': 10})
plt.yticks(np.arange(0.001, 0.035, 0.004))
plt.xticks(np.arange(0, 140, 20))
plt.ylabel('MDVP:Jitter(%)', fontsize='large', fontweight='bold')
plt.grid(lw = 0.35)

sns.set(style="whitegrid")
sns.boxplot(x='Persons', y='MDVP:Jitter(%)', data=df, palette="colorblind", hue='Class', 
            linewidth=1.5, dodge=False, order=["Healthy", "Patient"]
            ).set_title("Frequency Percentage Irregularity Ratio", fontweight='bold')
df.drop([99,149,189], inplace=True)

##########################################################
#MDVP:Jitter(Abs)
plt.scatter(x=index_class_h, y=df[df['Class']==1]["MDVP:Jitter(Abs)"][:139].sort_values(), 
            color='r', label="Patient")
plt.scatter(x=index_class_s, y=df[df['Class']==0]["MDVP:Jitter(Abs)"][:45].sort_values(), 
            color='b', label="Healthy")
plt.title('Frequency Absolute Irregularity Ratio(µs)', fontweight='bold', y=1.01, fontsize=12)
plt.legend(loc='upper left', bbox_to_anchor=(0.27, -0.06), ncol=2, prop={'size': 10})
plt.yticks(np.arange(0, 0.001, 0.1))
plt.xticks(np.arange(0, 141, 20))
plt.ylabel('MDVP:Jitter(Abs)', fontsize='large', fontweight='bold')
plt.grid(lw = 0.35)

sns.set(style="whitegrid")
sns.boxplot(x='Persons', y='MDVP:Jitter(Abs)', data=df, palette="colorblind", hue='Class', 
            linewidth=1.5, dodge=False, order=["Healthy", "Patient"]
            ).set_title("Frequency Absolute Irregularity Ratio(µs)", fontweight='bold')
df.drop(154, inplace=True)

##########################################################
#MDVP:RAP
plt.scatter(x=index_class_h, y=df[df['Class']==1]["MDVP:RAP"][:138].sort_values(), 
            color='r', label="Patient")
plt.scatter(x=index_class_s, y=df[df['Class']==0]["MDVP:RAP"][:45].sort_values(), 
            color='b', label="Healthy")
plt.title('Relative Average Perturbation', fontweight='bold', y=1.01, fontsize=12)
plt.legend(loc='upper left', bbox_to_anchor=(0.27, -0.06), ncol=2, prop={'size': 10})
plt.yticks(np.arange(0, 0.013, 0.001))
plt.xticks(np.arange(0, 141, 20))
plt.ylabel('MDVP:RAP', fontsize='large', fontweight='bold')
plt.grid(lw = 0.35)

sns.set(style="whitegrid")
sns.boxplot(x='Persons', y='MDVP:RAP', data=df, palette="colorblind", hue='Class', 
            linewidth=1.5, dodge=False, order=["Healthy", "Patient"]
            ).set_title("Relative Average Perturbation", fontweight='bold')

##########################################################
#Shimmer:APQ5 (Amplitude Perturbation Quotient)
plt.scatter(x=index_class_h, y=df[df['Class']==1]["Shimmer:APQ5"][:138].sort_values(), 
            color='r', label="Patient")
plt.scatter(x=index_class_s, y=df[df['Class']==0]["Shimmer:APQ5"][:45].sort_values(), 
            color='b', label="Healthy")
plt.title('Deviation in Five Amplitude', fontweight='bold', y=1.01, fontsize=12)
plt.legend(loc='upper left', bbox_to_anchor=(0.27, -0.06), ncol=2, prop={'size': 10})
plt.yticks(np.arange(0.005, 0.0551, 0.0025))
plt.xticks(np.arange(0, 141, 20))
plt.ylabel('Shimmer:APQ5', fontsize='large', fontweight='bold')
plt.grid(lw = 0.35)

sns.set(style="whitegrid")
sns.boxplot(x='Persons', y='Shimmer:APQ5', data=df, palette="colorblind", hue='Class', 
            linewidth=1.5, dodge=False, order=["Healthy", "Patient"]
            ).set_title("Deviation in Five Amplitude", fontweight='bold')

##########################################################
#MDVP:APQ
plt.scatter(x=index_class_h, y=df[df['Class']==1]["MDVP:APQ"][:138].sort_values(), 
            color='r', label="Patient")
plt.scatter(x=index_class_s, y=df[df['Class']==0]["MDVP:APQ"][:45].sort_values(), 
            color='b', label="Healthy")
plt.title('Deviation in Eleven Amplitude', fontweight='bold', y=1.01, fontsize=12)
plt.legend(loc='upper left', bbox_to_anchor=(0.27, -0.06), ncol=2, prop={'size': 10})
plt.yticks(np.arange(0.006, 0.0901, 0.0035))
plt.xticks(np.arange(0, 141, 20))
plt.ylabel('MDVP:APQ', fontsize='large', fontweight='bold')
plt.grid(lw = 0.35)

sns.set(style="whitegrid")
sns.boxplot(x='Persons', y='MDVP:APQ', data=df, palette="colorblind", hue='Class', 
            linewidth=1.5, dodge=False, order=["Healthy", "Patient"]
            ).set_title("Deviation in Eleven Amplitude", fontweight='bold')
df.drop(144, inplace=True)

##########################################################
#Shimmer:DDA
plt.scatter(x=index_class_h, y=df[df['Class']==1]["Shimmer:DDA"][:137].sort_values(), 
            color='r', label="Patient")
plt.scatter(x=index_class_s, y=df[df['Class']==0]["Shimmer:DDA"][:45].sort_values(), 
            color='b', label="Healthy")
plt.title('Difference of Differences of Amplitude', fontweight='bold', y=1.01, fontsize=12)
plt.legend(loc='upper left', bbox_to_anchor=(0.27, -0.06), ncol=2, prop={'size': 10})
plt.yticks(np.arange(0.01, 0.14, 0.005))
plt.xticks(np.arange(0, 141, 20))
plt.ylabel('Shimmer:DDA', fontsize='large', fontweight='bold')
plt.grid(lw = 0.35)

sns.set(style="whitegrid")
sns.boxplot(x='Persons', y='Shimmer:DDA', data=df, palette="colorblind", hue='Class', 
            linewidth=1.5, dodge=False, order=["Sağlıklı", "Hasta"]
            ).set_title("Difference of Differences of Amplitude", fontweight='bold')

##########################################################
#NHR
plt.scatter(x=index_class_h, y=df[df['Class']==1]["NHR"][:137].sort_values(), 
            color='r', label="Patient")
plt.scatter(x=index_class_s, y=df[df['Class']==0]["NHR"][:45].sort_values(), 
            color='b', label="Healthy")
plt.title('Noise to Harmonic Ratio', fontweight='bold', y=1.01, fontsize=12)
plt.legend(loc='upper left', bbox_to_anchor=(0.27, -0.06), ncol=2, prop={'size': 10})
plt.yticks(np.arange(0, 0.18, 0.010))
plt.xticks(np.arange(0, 141, 20))
plt.ylabel('NHR', fontsize='large', fontweight='bold')
plt.grid(lw = 0.35)

sns.set(style="whitegrid")
sns.boxplot(x='Persons', y='NHR', data=df, palette="colorblind", hue='Class', 
            linewidth=1.5, dodge=False, order=["Healthy", "Patient"]
            ).set_title("Noise to Harmonic Ratio", fontweight='bold')
df.drop([190, 98], inplace=True)

##########################################################
#HNR
plt.scatter(x=index_class_h, y=df[df['Class']==1]["HNR"][:136].sort_values(), 
            color='r', label="Patient")
plt.scatter(x=index_class_s, y=df[df['Class']==0]["HNR"][:44].sort_values(), 
            color='b', label="Healthy")
plt.title('Harmonic to Noise Ratio', fontweight='bold', y=1.01, fontsize=12)
plt.legend(loc='upper left', bbox_to_anchor=(0.27, -0.06), ncol=2, prop={'size': 10})
plt.yticks(np.arange(11, 34, 1))
plt.xticks(np.arange(0, 141, 20))
plt.ylabel('HNR', fontsize='large', fontweight='bold')
plt.grid(lw = 0.35)

sns.set(style="whitegrid")
sns.boxplot(x='Persons', y='HNR', data=df, palette="colorblind", hue='Class', 
            linewidth=1.5, dodge=False, order=["Healthy", "Patient"]
            ).set_title("Harmonic to Noise Ratio", fontweight='bold')

##########################################################
#RPDE
plt.scatter(x=index_class_h, y=df[df['Class']==1]["RPDE"][:136].sort_values(), 
            color='r', label="Patient")
plt.scatter(x=index_class_s, y=df[df['Class']==0]["RPDE"][:44].sort_values(), 
            color='b', label="Healthy")
plt.title('Recurrence Period Density Entropy', fontweight='bold', y=1.01, fontsize=12)
plt.legend(loc='upper left', bbox_to_anchor=(0.27, -0.06), ncol=2, prop={'size': 10})
plt.yticks(np.arange(0.25, 0.75, 0.05))
plt.xticks(np.arange(0, 141, 20))
plt.ylabel('RPDE', fontsize='large', fontweight='bold')
plt.grid(lw = 0.35)

sns.set(style="whitegrid")
sns.boxplot(x='Persons', y='RPDE', data=df, palette="colorblind", 
            hue='Class', linewidth=1.5, dodge=False,
            order=["Healthy", "Patient"]
            ).set_title("Recurrence Period Density Entropy", fontweight='bold')

##########################################################
#D2
plt.scatter(x=index_class_h, y=df[df['Class']==1]["D2"][:136].sort_values(), 
            color='r', label="Patient")
plt.scatter(x=index_class_s, y=df[df['Class']==0]["D2"][:44].sort_values(), 
            color='b', label="Healthy")
plt.title('Dimension', fontweight='bold', y=1.01, fontsize=12)
plt.legend(loc='upper left', bbox_to_anchor=(0.27, -0.06), ncol=2, prop={'size': 10})
plt.yticks(np.arange(1.3, 3.71, 0.2))
plt.xticks(np.arange(0, 141, 20))
plt.ylabel('D2', fontsize='large', fontweight='bold')
plt.grid(lw = 0.35)

sns.set(style="whitegrid")
sns.boxplot(x='Persons', y='D2', data=df, palette="colorblind", hue='Class', 
            linewidth=1.5, dodge=False, order=["Healthy", "Patient"]
            ).set_title("Dimension", fontweight='bold')

##########################################################
#DFA
plt.scatter(x=index_class_h, y=df[df['Class']==1]["DFA"][:136].sort_values(), 
            color='r', label="Patient")
plt.scatter(x=index_class_s, y=df[df['Class']==0]["DFA"][:44].sort_values(), 
            color='b', label="Healthy")
plt.title('Detrended Fluctuation Analysis', fontweight='bold', y=1.01, fontsize=12)
plt.legend(loc='upper left', bbox_to_anchor=(0.27, -0.06), ncol=2, prop={'size': 10})
plt.yticks(np.arange(0.57, 0.84, 0.01))
plt.xticks(np.arange(0, 141, 20))
plt.ylabel('DFA',fontsize='large', fontweight='bold')
plt.grid(lw = 0.35)

sns.set(style="whitegrid")
sns.boxplot(x='Persons', y='DFA', data=df, palette="colorblind", hue='Class', 
            linewidth=1.5, dodge=False, order=["Healthy", "Patient"]
            ).set_title("Detrended Fluctuation Analysis", fontweight='bold')

##########################################################
#Spread1
plt.scatter(x=index_class_h, y=df[df['Class']==1]["Spread1"][:136].sort_values(), 
            color='r', label="Patient")
plt.scatter(x=index_class_s, y=df[df['Class']==0]["Spread1"][:44].sort_values(), 
            color='b', label="Healthy")
plt.title('Frequency Change Measure 1', fontweight='bold', y=1.01, fontsize=12)
plt.legend(loc='upper left', bbox_to_anchor=(0.27, -0.06), ncol=2, prop={'size': 10})
plt.yticks(np.arange(-8, -2.4, 0.3))
plt.xticks(np.arange(0, 141, 20))
plt.ylabel('Spread1', fontsize='large', fontweight='bold')
plt.grid(lw = 0.35)

sns.set(style="whitegrid")
sns.boxplot(x='Persons', y='Spread1', data=df, palette="colorblind", hue='Class', 
            linewidth=1.5, dodge=False, order=["Healthy", "Patient"]
            ).set_title("Frequency Change Measure 1", fontweight='bold')

##########################################################
#Spread2
plt.scatter(x=index_class_h, y=df[df['Class']==1]["Spread2"][:136].sort_values(), 
            color='r', label="Patient")
plt.scatter(x=index_class_s, y=df[df['Class']==0]["Spread2"][:44].sort_values(), 
            color='b', label="Healthy")
plt.title('Frequency Change Measure 2', fontweight='bold', y=1.01, fontsize=12)
plt.legend(loc='upper left', bbox_to_anchor=(0.27, -0.06), ncol=2, prop={'size': 10})
plt.yticks(np.arange(0, 0.45, 0.02))
plt.xticks(np.arange(0, 141, 20))
plt.ylabel('Spread2', fontsize='large', fontweight='bold')
plt.grid(lw = 0.35)

sns.set(style="whitegrid")
sns.boxplot(x='Persons', y='Spread2', data=df, palette="colorblind", hue='Class', 
            linewidth=1.5, dodge=False, order=["Healthy", "Patient"]
            ).set_title("Frequency Change Measure 2", fontweight='bold')

##########################################################
#PPE
plt.scatter(x=index_class_h, y=df[df['Class']==1]["PPE"][:136].sort_values(), 
            color='r', label="Patient")
plt.scatter(x=index_class_s, y=df[df['Class']==0]["PPE"][:44].sort_values(), 
            color='b', label="Healthy")
plt.title('Pitch Period Entropy', fontweight='bold', y=1.01, fontsize=12)
plt.legend(loc='upper left', bbox_to_anchor=(0.27, -0.06), ncol=2, prop={'size': 10})
plt.yticks(np.arange(0.04, 0.45, 0.02))
plt.xticks(np.arange(0, 141, 20))
plt.ylabel('PPE', fontsize='large', fontweight='bold')
plt.grid(lw = 0.35)

sns.set(style="whitegrid")
sns.boxplot(x='Persons', y='PPE', data=df, palette="colorblind", hue='Class', 
            linewidth=1.5, dodge=False, order=["Healthy", "Patient"]
            ).set_title("Pitch Period Entropy", fontweight='bold')

##########################################################
#Pairplots
#Pairplots viewed for type of the distributions
#Some skew variables was transformed with logarithm method
sns.pairplot(df.iloc[:,0:9], diag_kind="kde")
df.iloc[:,0:9] = np.log(df.iloc[:,0:9])
sns.pairplot(df.iloc[:,0:9], diag_kind="kde")

sns.pairplot(df.iloc[:,9:17], diag_kind="kde")
df.iloc[:,[9,15,16]] = np.log(df.iloc[:,[9,15,16]])
sns.pairplot(df.iloc[:,9:17], diag_kind="kde")

##########################################################
#Normality curves was drawn so that whether the distributions approach to normal
from scipy.stats import norm
from scipy import stats

#Column names
columns=[]
for col in df.columns:
    columns.append(col)
    
for i, col in enumerate(df.iloc[:,0:6].columns):
    plt.subplot(2, 6, i+1) 
    sns.distplot(df[col], fit=norm)
    plt.xlabel(col, fontweight='bold')
    plt.tight_layout()   
    for i, col in enumerate(df.iloc[:,0:6].columns):
        plt.subplot(2, 6, i+7) 
        res = stats.probplot(df[col], plot=plt) 

##########################################################       
for i, col in enumerate(df.iloc[:,6:12].columns):
    plt.subplot(2, 6, i+1) 
    sns.distplot(df[col], fit=norm)
    plt.xlabel(col, fontweight='bold')
    plt.tight_layout()   
    for i, col in enumerate(df.iloc[:,6:12].columns):
        plt.subplot(2, 6, i+7) 
        res = stats.probplot(df[col], plot=plt)
        
##########################################################
for i, col in enumerate(df.iloc[:,12:17].columns):
    plt.subplot(2, 5, i+1) 
    sns.distplot(df[col], fit=norm)
    plt.xlabel(col, fontweight='bold')
    plt.tight_layout()   
    for i, col in enumerate(df.iloc[:,12:17].columns):
        plt.subplot(2, 5, i+6) 
        res = stats.probplot(df[col], plot=plt)    
        
#################################################################################################
#NORMALIZATION
df = (df-np.min(df))/(np.max(df)-np.min(df))

##########################################################
#Due to some values eliminate and do normalization, index edited and last version saved
df.reset_index(inplace=True)
df.drop("index", 1, inplace=True)
df.drop(["Persons"], 1, inplace=True)
df.to_csv("parkin_son.csv", index=False)
df = pd.read_csv("parkin_son.csv")

#################################################################################################
#MACHINE LEARNING

#Determine the dependent and independent variables
x= df.iloc[:,:-1].values #independent
y=df.iloc[:,-1:].values #dependent 

#Holdout method
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
print('x_train', x_train.shape)
print('x_test', x_test.shape)
print('y_train', y_train.shape)
print('y_test', y_test.shape)

#KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

parameters_knn = {'n_neighbors':[3,4,5,6,7,8,9,10],
              'weights':['uniform', 'distance'],
              'metric':['euclidean','manhattan', 'minkowski'],
              'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
              'n_jobs':[-1]
             }

gs=GridSearchCV(
        KNeighborsClassifier(random_state=0),
        parameters_knn,
        verbose=1,
        cv=10,
        n_jobs=-1,
        scoring='roc_auc'
        )

gs.fit(x_train, y_train)
print(gs.best_params_)

knn_pred=gs.predict(x_test)

from sklearn.metrics import accuracy_score
print("Accuracy:{:.2%}".format(accuracy_score(y_test, knn_pred)))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, knn_pred)

plt.clf()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.GnBu)
classNames = ['Healthy','Patient']
plt.title('KNN Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))

#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression

parameters_lr = {
    'class_weight' : ['balanced', None, {1:0.5, 0:0.5}, {1:0.4, 0:0.6}, {1:0.6, 0:0.4}, {1:0.7, 0:0.3}], 
    'penalty' : ['l1', 'l2'],
    'C' : [0.01,0.1,1,10,100],
    'solver' : ['liblinear', 'saga']
             }

gs2 = GridSearchCV(LogisticRegression(random_state=0), 
                   parameters_lr, 
                   cv = 10, 
                   verbose=1, 
                   scoring='roc_auc',
                   n_jobs=-1
                  )

gs2.fit(x_train, y_train)
print(gs2.best_params_)

lr_pred=gs2.predict(x_test)

print("Accuracy:{:.2%}".format(accuracy_score(y_test, lr_pred)))

cm2 = confusion_matrix(y_test, lr_pred)

plt.clf()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.GnBu)
classNames = ['Healthy','Patient']
plt.title('Logistic Regression Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
