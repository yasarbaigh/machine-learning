# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC



# Load dataset
#url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
url = "dataset/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)


# shape
print ('\n---------------------------------')
print ('data- structure')

print(dataset.shape)



# head
print ('\n---------------------------------')
print ('limit 20 records')

print(dataset.head(20))

print ('\n---------------------------------')
print ('Stnd Numbers')
print(dataset.describe())



print ('\n---------------------------------')
print ('class distribution')
print(dataset.groupby('class').size())


print ('\n---------------------------------')
print ('univariate plots')
# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()



print ('\n---------------------------------')
print ('histograms plots')

dataset.hist()
plt.show()

# scatter plot matrix
scatter_matrix(dataset)
plt.show()

