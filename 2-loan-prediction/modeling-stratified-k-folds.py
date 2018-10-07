#https://trainings.analyticsvidhya.com/courses/course-v1:AnalyticsVidhya+LP101+2018_T1/courseware/
# Modeling data-sets

# Load libraries

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

import pandas as pd
import numpy as np                     # For mathematical calculations
import seaborn as sns                  # For data visualization
import matplotlib.pyplot as plt        # For plotting graphs
#%matplotlib inline
import warnings                        # To ignore any warnings
warnings.filterwarnings("ignore")

# function to display distribution graph, 
def showPlot(dataset, field):
	plt.figure(1)
	plt.subplot(121)
	df=dataset.dropna()
	sns.distplot(df[field]);

	plt.subplot(122)
	dataset[field].plot.box(figsize=(16,5))
	plt.show()

#function for inline replace with mode values of the columns 
def replace_with_mode(dataset, fields):
	for item in fields:			
		dataset[item].fillna(dataset[item].mode()[0], inplace=True)

#Reading data
train=pd.read_csv("dataset/train_u6lujuX_CVtuZ9i.csv")
test=pd.read_csv("dataset/test_Y3wMUE5_7gLdaTN.csv")


train_original=train.copy()
test_original=test.copy()


#replacing null/nan values with mode values
replace_with_mode(train, ['Credit_History', 'Loan_Amount_Term', 'LoanAmount'])
replace_with_mode(test, ['Credit_History', 'Loan_Amount_Term', 'LoanAmount'])

print ('\n\n---------------------------')
print('After nan removal opertaion')
print(train.isnull().any())


#since loan-id, no link with loan-approval process, so removing it from the process
print ('\n\n---------------------------')
print('Dropping loan-id from dataset')
train=train.drop('Loan_ID',axis=1)
test=test.drop('Loan_ID',axis=1)


#skilearn  for modelling
#as skilearn needs target-variable in separate data-set, so removing loan-status from train-data-set
X = train.drop('Loan_Status',1)
y = train.Loan_Status


# logistic-Regression needs all data in numerical format,so gender,degree,location,
# will be replaced with numerical values as 0 0r 1 , 
X=pd.get_dummies(X)
train=pd.get_dummies(train)
test=pd.get_dummies(test)



print ('\n\n---------------------------')
print('Model cross validation with StratifiedKFold of 5')


i=1
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)

# stratified k-fold cross validation with 5 folds.
for train_index,test_index in kf.split(X,y):
     print('\n{} of kfold {}'.format(i,kf.n_splits))
     xtr,xvl = X.loc[train_index],X.loc[test_index]
     ytr,yvl = y[train_index],y[test_index]
    
     model = LogisticRegression(random_state=1)
     model.fit(xtr, ytr)
     pred_test = model.predict(xvl)
     score = accuracy_score(yvl,pred_test)
     print('accuracy_score',score)
     i+=1

pred_test = model.predict(test)
pred=model.predict_proba(xvl)[:,1]



print ('\n\n---------------------------')


print ('\n\n---------------------------')
print('')

print ('\n\n---------------------------')
print('')
