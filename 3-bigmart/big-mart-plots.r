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



#since target variable in Item_outelet_sales
#viewing train data's value currently
ggplot(train) + geom_histogram(aes(train$Item_Outlet_Sales), binwidth = 100, fill = "darkgreen") + xlab("Item_Outlet_Sales")


#plots of continous data  
p1 = ggplot(combi) + geom_histogram(aes(Item_Weight), binwidth = 0.5, fill = "blue")
p2 = ggplot(combi) + geom_histogram(aes(Item_Visibility), binwidth = 0.005, fill = "blue")
p3 = ggplot(combi) + geom_histogram(aes(Item_MRP), binwidth = 1, fill = "blue")
plot_grid(p1, p2, p3, nrow = 1) # plot_grid() from cowplot package

#plots of category data
ggplot(combi %>% group_by(Item_Fat_Content) %>% summarise(Count = n())) + 
  geom_bar(aes(Item_Fat_Content, Count), stat = "identity", fill = "coral1")


#since lf,lowfat,Low Fat are same  and reg and Regular are same grouping the combined data.
combi$Item_Fat_Content[combi$Item_Fat_Content == "LF"] = "Low Fat"
combi$Item_Fat_Content[combi$Item_Fat_Content == "low fat"] = "Low Fat"
combi$Item_Fat_Content[combi$Item_Fat_Content == "reg"] = "Regular"
ggplot(combi %>% group_by(Item_Fat_Content) %>% summarise(Count = n())) + 
  geom_bar(aes(Item_Fat_Content, Count), stat = "identity", fill = "coral1")




#Plost of  other categorical variables.
# plot for Item_Type
p4 = ggplot(combi %>% group_by(Item_Type) %>% summarise(Count = n())) + 
  geom_bar(aes(Item_Type, Count), stat = "identity", fill = "coral1") +
  xlab("") +
  geom_label(aes(Item_Type, Count, label = Count), vjust = 0.5) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  ggtitle("Item_Type")

# plot for Outlet_Identifier
p5 = ggplot(combi %>% group_by(Outlet_Identifier) %>% summarise(Count = n())) + 
  geom_bar(aes(Outlet_Identifier, Count), stat = "identity", fill = "coral1") +
  geom_label(aes(Outlet_Identifier, Count, label = Count), vjust = 0.5) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# plot for Outlet_Size
p6 = ggplot(combi %>% group_by(Outlet_Size) %>% summarise(Count = n())) + 
  geom_bar(aes(Outlet_Size, Count), stat = "identity", fill = "coral1") +
  geom_label(aes(Outlet_Size, Count, label = Count), vjust = 0.5) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

second_row = plot_grid(p5, p6, nrow = 1)
plot_grid(p4, second_row, ncol = 1)

# Outlet plot shows that 4k  records are blank,
# we will chk this  by bi-variate analysis

# plot for Outlet_Establishment_Year
p7 = ggplot(combi %>% group_by(Outlet_Establishment_Year) %>% summarise(Count = n())) + 
  geom_bar(aes(factor(Outlet_Establishment_Year), Count), stat = "identity", fill = "coral1") +
  geom_label(aes(factor(Outlet_Establishment_Year), Count, label = Count), vjust = 0.5) +
  xlab("Outlet_Establishment_Year") +
  theme(axis.text.x = element_text(size = 8.5))


# plot for Outlet_Type
p8 = ggplot(combi %>% group_by(Outlet_Type) %>% summarise(Count = n())) + 
  geom_bar(aes(Outlet_Type, Count), stat = "identity", fill = "coral1") +
  geom_label(aes(factor(Outlet_Type), Count, label = Count), vjust = 0.5) +
  theme(axis.text.x = element_text(size = 8.5))


# ploting both plots together
plot_grid(p7, p8, ncol = 2)
#Observations


#from plots year 1998 has lesser outlets as compared to the other years.
#Supermarket Type 1 seems to be the most popular category of Outlet_Type.

"
-----------------------------------------------------------
Bivariate Analysis

 discover hidden relationships between the independent variable and the target variable and use those findings in missing data imputation and feature engineering in the next module
"

"scatter plots for the continuous or numeric variables
 and violin plots for the categorical variables."
 
#scatter plots for continous values with target(item-sales)
train = combi[1:nrow(train)] # extracting train data from the combined data

# Item_Weight vs Item_Outlet_Sales
p9 = ggplot(train) + 
     geom_point(aes(Item_Weight, Item_Outlet_Sales), colour = "violet", alpha = 0.3) +
     theme(axis.title = element_text(size = 8.5))
	 
# Item_Visibility vs Item_Outlet_Sales
p10 = ggplot(train) + 
      geom_point(aes(Item_Visibility, Item_Outlet_Sales), colour = "violet", alpha = 0.3) +
      theme(axis.title = element_text(size = 8.5))
	  
# Item_MRP vs Item_Outlet_Sales
p11 = ggplot(train) + 
      geom_point(aes(Item_MRP, Item_Outlet_Sales), colour = "violet", alpha = 0.3) +
      theme(axis.title = element_text(size = 8.5))
	  
	  
second_row_2 = plot_grid(p10, p11, ncol = 2)
plot_grid(p9, second_row_2, nrow = 2)
"
Observations

Item_Outlet_Sales is spread well across the entire range of the Item_Weight without any obvious pattern.
In Item_Visibility vs Item_Outlet_Sales, there is a string of points at Item_Visibility = 0.0 which seems strange as item visibility cannot be completely zero. We will take note of this issue and deal with it in the later stages.
In the third plot of Item_MRP vs Item_Outlet_Sales, we can clearly see 4 segments of prices that can be used in feature engineering to create a new variable.

"


#violin plots for categorical values with target(item-sales)

# Item_Type vs Item_Outlet_Sales
p12 = ggplot(train) + 
      geom_violin(aes(Item_Type, Item_Outlet_Sales), fill = "magenta") +
      theme(axis.text.x = element_text(angle = 45, hjust = 1),
            axis.text = element_text(size = 6),
            axis.title = element_text(size = 8.5))

# Item_Fat_Content vs Item_Outlet_Sales
p13 = ggplot(train) + 
      geom_violin(aes(Item_Fat_Content, Item_Outlet_Sales), fill = "magenta") +
      theme(axis.text.x = element_text(angle = 45, hjust = 1),
            axis.text = element_text(size = 8),
            axis.title = element_text(size = 8.5))

# Outlet_Identifier vs Item_Outlet_Sales
p14 = ggplot(train) + 
      geom_violin(aes(Outlet_Identifier, Item_Outlet_Sales), fill = "magenta") +
      theme(axis.text.x = element_text(angle = 45, hjust = 1),
            axis.text = element_text(size = 8),
            axis.title = element_text(size = 8.5))


second_row_3 = plot_grid(p13, p14, ncol = 2)
plot_grid(p12, second_row_3, ncol = 1)
"
Observations

Distribution of Item_Outlet_Sales across the categories of Item_Type is not very distinct and same is the case with Item_Fat_Content.

The distribution for OUT010 and OUT019 categories of Outlet_Identifier are quite similar and very much different from the rest of the categories of Outlet_Identifier.

"	

#since outlet size has blank data, and blank-data count is similar to small outlet-size count,
# replacing blank with small-size values

p15 = ggplot(train) + geom_violin(aes(Outlet_Location_Type, Item_Outlet_Sales), fill = "magenta")
p16 = ggplot(train) + geom_violin(aes(Outlet_Type, Item_Outlet_Sales), fill = "magenta")
plot_grid(p15, p16, ncol = 1)


		
print("last-line")




