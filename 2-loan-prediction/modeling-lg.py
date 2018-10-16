#https://trainings.analyticsvidhya.com/courses/course-v1:AnalyticsVidhya+LP101+2018_T1/courseware/
# Modeling data-sets

# Load libraries

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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
print('X,y for logistic-Regression')
print(X.isnull().any())
print(y.isnull().any())



#train dataset, is spiltted in 2 sets, train and validate, 
#so that we can validate the train, is worthy for predicting in test-data-set
x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size =0.3)

model = LogisticRegression()
# assiging the train set, for regression
model.fit(x_train, y_train)



pred_cv = model.predict(x_cv)
print ('\n\n---------------------------')
print('accuracy_score is closer to 80% for train-validation-set ')
print (str(accuracy_score(y_cv, pred_cv)))





print ('\n\n---------------------------')
print('predicting for test-data-set, and accuracy_score is')
pred_test = model.predict(test)

submission=pd.read_csv("dataset/Sample_Submission_ZAuTl8O_FK3zQHh.csv")
submission['Loan_Status']=pred_test
submission['Loan_ID']=test_original['Loan_ID']


submission['Loan_Status'].replace(0, 'N',inplace=True)
submission['Loan_Status'].replace(1, 'Y',inplace=True)
pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('output/logistic.csv')


