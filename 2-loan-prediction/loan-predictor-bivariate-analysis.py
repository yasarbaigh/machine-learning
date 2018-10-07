#https://trainings.analyticsvidhya.com/courses/course-v1:AnalyticsVidhya+LP101+2018_T1/courseware/
# Bi variate Analysis

# Load libraries

import pandas as pd
import numpy as np                     # For mathematical calculations
import seaborn as sns                  # For data visualization
import matplotlib.pyplot as plt        # For plotting graphs
#%matplotlib inline
import warnings                        # To ignore any warnings
warnings.filterwarnings("ignore")

#Reading data
train=pd.read_csv("dataset/train_u6lujuX_CVtuZ9i.csv")
test=pd.read_csv("dataset/test_Y3wMUE5_7gLdaTN.csv")


train_original=train.copy()
test_original=test.copy()


# Print data types for each variable
print ('---------------------------\nTrain data-types')
print(train.dtypes)

print ('\n---------------------------\nTest data-types')
print(test.dtypes)

print ('\n---------------------------\nTrain count records')
print(train.shape)

print ('\n---------------------------\nTest count records')
print(test.shape)


print ('\n---------------------------\nTrain Loan status')
print(train['Loan_Status'].value_counts())

# Normalize can be set to True to print proportions instead of number 
print ('\n---------------------------\nTrain Loan in percentage is achieved by normalize')
print(train['Loan_Status'].value_counts(normalize=True))



print('Visulaizing categorical variables (self-employed, graduate etc) vs target variable (loan approval status).')
Gender=pd.crosstab(train['Gender'],train['Loan_Status'])
Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))

# TBD
#<matplotlib.axes._subplots.AxesSubplot at 0x7ff804e15eb8>

Married=pd.crosstab(train['Married'],train['Loan_Status'])
Dependents=pd.crosstab(train['Dependents'],train['Loan_Status'])
Education=pd.crosstab(train['Education'],train['Loan_Status'])
Self_Employed=pd.crosstab(train['Self_Employed'],train['Loan_Status'])

Married.div(Married.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.show()

Dependents.div(Dependents.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.show()

Education.div(Education.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.show()

Self_Employed.div(Self_Employed.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.show()



Credit_History=pd.crosstab(train['Credit_History'],train['Loan_Status'])
Property_Area=pd.crosstab(train['Property_Area'],train['Loan_Status'])

Credit_History.div(Credit_History.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.show()

Property_Area.div(Property_Area.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.show()




print('Numerical Independent Variable vs Target Variable')
# mean income of people for which the loan has been approved
# vs the mean income of people for which the loan has not been approved.
train.groupby('Loan_Status')['ApplicantIncome'].mean().plot.bar()
plt.show()


print('\n\n\n------------------------------------------')
print('Categorizing income group into low, high medium to check loan approval')
bins=[0,2500,4000,6000,81000]
group=['Low','Average','High', 'Very high']
train['Income_bin']=pd.cut(train['ApplicantIncome'],bins,labels=group)


Income_bin=pd.crosstab(train['Income_bin'],train['Loan_Status'])
Income_bin.div(Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('ApplicantIncome')
P = plt.ylabel('Percentage')
plt.show()

#It can be inferred that Applicant income does not affect the chances of loan approval 
#which contradicts our hypothesis 
#in which we assumed that if the applicant income is high the chances of loan approval will also be high.

print('\n\n\n------------------------------------------')
print('Categorizing Coapplicant_Income_bin into low, high medium to check loan approval')

bins=[0,1000,3000,42000]
group=['Low','Average','High']
train['Coapplicant_Income_bin']=pd.cut(train['CoapplicantIncome'],bins,labels=group)

co_app_Income_bin=pd.crosstab(train['Coapplicant_Income_bin'],train['Loan_Status'])
co_app_Income_bin.div(co_app_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('Coapplicant_Income_bin')
P = plt.ylabel('Percentage')
plt.show()

'''
It shows that if coapplicant’s income is less the chances of loan approval are high.
 But this does not look right. The possible reason behind this may be that most of 
 the applicants don’t have any coapplicant so the coapplicant income for such 
 applicants is 0 and hence the loan approval is not dependent on it.
'''


print('\n\n\n------------------------------------------')
print('Add Applicant Income and Coapplicant Income to see the combined effect of Total Income on the Loan_Status approval')
train['total_income']=train['ApplicantIncome'] + train['CoapplicantIncome']


bins=[0,2500,4000,6000,81000]
group=['Low','Average','High', 'Very high']
train['total_Income_bin']=pd.cut(train['total_income'],bins,labels=group)


total_Income_bin=pd.crosstab(train['total_Income_bin'],train['Loan_Status'])
total_Income_bin.div(total_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('TotalApplicantIncome')
P = plt.ylabel('Percentage')
plt.show()


'''
We can see that Proportion of loans getting approved for applicants having low 
Total_Income is very less as compared to that of applicants with Average,
High and Very High Income

'''


print('\n\n\n------------------------------------------')
print('Compairng loan amount with Loan_Status approval')
bins=[0,100,200,700]
group=['Low','Average','High']
train['LoanAmount_bin']=pd.cut(train['LoanAmount'],bins,labels=group)


LoanAmount_bin=pd.crosstab(train['LoanAmount_bin'],train['Loan_Status'])
LoanAmount_bin.div(LoanAmount_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('LoanAmount_bin')
P = plt.ylabel('Percentage')
plt.show()




print('\n\n\n------------------------------------------')
print('Types and shape of Train data and dropping additional bins')
print(train.dtypes)
print(train.shape)

train=train.drop(['Income_bin', 'Coapplicant_Income_bin', 'LoanAmount_bin', 'total_Income_bin', 'total_income'], axis=1)

print('After dropping additional columns')
print(train.dtypes)
print(train.shape)


print('\n\n\n------------------------------------------')
print('Replacing dependent, Loan_Status with Numerical values for correlation heat-map')

train['Dependents'].replace('3+', 3,inplace=True)
test['Dependents'].replace('3+', 3,inplace=True)
train['Loan_Status'].replace('N', 0,inplace=True)
train['Loan_Status'].replace('Y', 1,inplace=True)

matrix = train.corr()
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(matrix, vmax=.8, square=True, cmap="BuPu");
plt.show()

print('\n\n\n------------------------------------------')
print('')


print('\n\n\n------------------------------------------')
print('')
