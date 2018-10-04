# Load libraries

import pandas as pd
import numpy as np                     # For mathematical calculations
import seaborn as sns                  # For data visualization
import matplotlib.pyplot as plt        # For plotting graphs
#%matplotlib inline
import warnings                        # To ignore any warnings
#warnings.filterwarnings("ignore")

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


plt.figure(1)
plt.subplot(221)
train['Gender'].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'Gender')

plt.subplot(222)
train['Married'].value_counts(normalize=True).plot.bar(title= 'Married')

plt.subplot(223)
train['Self_Employed'].value_counts(normalize=True).plot.bar(title= 'Self_Employed')

plt.subplot(224)
train['Credit_History'].value_counts(normalize=True).plot.bar(title= 'Credit_History')


plt.show()



plt.figure(1)
plt.subplot(131)
train['Dependents'].value_counts(normalize=True).plot.bar(figsize=(24,6), title= 'Dependents')

plt.subplot(132)
train['Education'].value_counts(normalize=True).plot.bar(title= 'Education')

plt.subplot(133)
train['Property_Area'].value_counts(normalize=True).plot.bar(title= 'Property_Area')

plt.show()



