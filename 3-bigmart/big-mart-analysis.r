library(data.table) # used for reading and manipulation of data
library(dplyr)      # used for data manipulation and joining
library(ggplot2)    # used for ploting 
library(caret)      # used for modeling
library(corrplot)   # used for making correlation plot
library(xgboost)    # used for building XGBoost model
library(cowplot)    # used for combining multiple plots 

train = fread("D:/datasets/R/3-bigmart/Train_UWu5bXk.csv") 
test = fread("D:/datasets/R/3-bigmart/Test_u94Q5KV.csv")
#submission = fread("dataset/SampleSubmission_TmnO39y.csv")

# rows and colummns
dim(train);dim(test);

print("Train-DAta Columns")
names(train);
print("Test-DAta Columns")
names(test);

print("\n\n train-DAta summary")
str(train)
print("\n\n test-DAta summary")
str(test)

# combining train and test for easy computation
test[,Item_Outlet_Sales := NA]
combi = rbind(train, test) # combining train and test datasets
dim(combi)

#since lf,lowfat,Low Fat are same  and reg and Regular are same grouping the combined data.
combi$Item_Fat_Content[combi$Item_Fat_Content == "LF"] = "Low Fat"
combi$Item_Fat_Content[combi$Item_Fat_Content == "low fat"] = "Low Fat"
combi$Item_Fat_Content[combi$Item_Fat_Content == "reg"] = "Regular"

#Outlet size has blank values




print("last-line")



