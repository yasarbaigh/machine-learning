#https://trainings.analyticsvidhya.com/courses/course-v1:AnalyticsVidhya+LP101+2018_T1/courseware/
# Missing Values

# Load libraries

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


#Reading data
train=pd.read_csv("dataset/train_u6lujuX_CVtuZ9i.csv")
test=pd.read_csv("dataset/test_Y3wMUE5_7gLdaTN.csv")


train_original=train.copy()
test_original=test.copy()


print ('\n\n---------------------------')
print('Missing values count in dataset')
print(train.isnull().sum())


'''

Simple methods to fill the missing values:

For numerical variables: imputation using mean or median
For categorical variables: imputation using mode

'''
# categorical values filled with mode
train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)

#though its numerical value, filling with mode
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)


'''
 LoanAmount variable. As it is a numerical variable, we can use mean or median
  to impute the missing values. We will use median to fill the null values 
  as earlier we saw that loan amount have outliers so the mean will not be 
  the proper approach as it is highly affected by the presence of outliers.

'''
train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)


print ('\n\n---------------------------')
print('Filled with train-data median and mode values, checking is null in dataset')
print(train.isnull().sum())



# Filling the test-data with mean, median and mode values
test['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
test['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
test['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
test['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
test['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
test['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)

print ('\n\n---------------------------')
print('Filled with Test-data median and mode values, checking is null in dataset')
print(test.isnull().sum())






print ('\n\n---------------------------')
print('')

print(' Distribution of LoanAmount variable check skewness.')
showPlot(train, 'LoanAmount')


# To avoid, skewness, in loan-amount ,applying log-transformation in train and test dataset
print(' Applied log-transformation in LoanAmount of test-data and train-data.')

train['LoanAmount_log'] = np.log(train['LoanAmount'])
train['LoanAmount_log'].hist(bins=20)
test['LoanAmount_log'] = np.log(test['LoanAmount'])

showPlot(train, 'LoanAmount_log')




print ('\n\n---------------------------')
print('')



print ('\n\n---------------------------')
print('')


print ('\n\n---------------------------')
print('')

