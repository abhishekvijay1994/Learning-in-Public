#!/usr/bin/env python
# coding: utf-8

# In[4]:


#invite people for the Kaggle party
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[38]:


df_train = pd.read_csv('/Users/abhishekvijay/Documents/kaggle/housing/train.csv')
#imports the file path 


# In[6]:


df_train.columns
#displays the columns in the data


# In[7]:


df_train['SalePrice'].describe()
#describes the data with a series of output that will important to understand the data


# In[92]:


#Theres no 0 values that would potentially ruin the model


# In[8]:


sns.distplot(df_train['SalePrice'])
#using seaborn to display a distrobution plot


# In[38]:


#deviates from the normal distribution
#positive skewness: tail end on the right side is longer
#has a peak


# In[9]:


print("skewness: %f" % df_train['SalePrice'].skew())
print("kurtosis: %f" % df_train['SalePrice'].kurt())
#display skewness, still need to figure out %f meaning fully


# In[40]:


#skewness is between >1 which means the data is highly skewed

#kurtosis is >3 AKA leptokurtic which means the data is heavy tailed with outliers shown by
#the narrowness of the right tail end


# In[10]:


#lets see the relationship between saleprice and ground living room area

var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'],df_train[var]], axis = 1)
#concat used to merge data from saleprice and grlivearea
data.plot.scatter(x=var,y='SalePrice', ylim=(0,800000))
#taking the merged data aka var and displaying it as scatter plot 
#by setting the x and y axis variables and setting a limit on the y axis


# In[42]:


#there is a linear relationship between saleprice and GrLivArea

#lets see the relationship between saleprice and total basement square foot TotalBsmntSF


# In[11]:


var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'],df_train[var]], axis=1)
data.plot.scatter(x=var,y='SalePrice',ylim=(0,800000))


# In[44]:


#strong linear relationship based on the sudden exponential increase
#looks like having ~ 500-1500 sq doesnt impact salesprice

#lets conduct analysis on the categorical features (not numerical descriptive traits)


# In[12]:


var = 'OverallQual'
#define OverallQual as the variable being used
data = pd.concat([df_train['SalePrice'],df_train[var]],axis = 1)
#concat the data of salesprice and overall quality using pandas 
f, ax = plt.subplots(figsize=(8,6))
#plt.sublot is a function that returns a tuple containing a figure and axes object(s)
#tuple is unpacked into the f and ax variables
fig = sns.boxplot(x=var,y='SalePrice', data=data)
#created the boxplot
fig.axis(ymin=0,ymax=800000)
#setting the axis size


# In[46]:


#looks like salesprice is driven up by the overall quality

#lets see how year built looks


# In[13]:


var = 'YearBuilt'
data = pd.concat([df_train['SalePrice'],df_train[var]], axis = 1)
f, ax = plt.subplots(figsize = (16,8))
fig = sns.boxplot (x=var,y='SalePrice',data=data)
fig.axis(ymin=0,ymax=800000)
plt.xticks(rotation=90)
#use this function to rotate the tick labels 90 degrees to see labels clearly


# In[48]:


#Sales price is driven by the year built although its not a strong relationship


# In[49]:


#Summary

#Ground living room area and total basement square footage seem to be linearly related with Sale price. 
#Both are positively related, as one variable increases, the other also increases
#Overall quality and year built also show a relationship with salesprice, although it is not as strong with year built

#lets analyze more
#correlation matrix (heatmap - seaborn)
#salesprice correlation matrix (zoomed heatmap)
#scatter plots between correlated variables


# In[14]:


#correlation matrix
corrmat = df_train.corr()
f, ax = plt.subplots(figsize = (12,9))
sns.heatmap(corrmat, vmax = .8, square=True)


# In[51]:


#By first looking at this correlation matrix, 2 parts stand out:
# "total bsmt sf" and '1stFlrSF'
#the Garage x variables
#there is a high correlation between these variables which may be a case of multicollinearity
#if you think about these variables, we can see that multicollinearity is occuring
#basement size will usually be the size of the first floor and the bigger the garage size, the more cars
#salesprice has a few squares that stand out: OverallQual, GrLivArea and Total BsmtSF

#lets take a look at the salesprice correlation matrix (zoomed heatmap style) next


# In[15]:


k = 10
#number of variables for the heatmap
cols = corrmat.nlargest(k,'SalePrice')['SalePrice'].index
#getting the 10 largest correlated variables and getting their row labels
cm = np.corrcoef(df_train[cols].values.T)
#transposing values of "cols"  into correlation coefficients
sns.set(font_scale=1.25)
#set font fize 
hm = sns.heatmap(cm, cbar=True,annot=True,square=True, fmt='.2f',annot_kws={'size':10}, yticklabels=cols.values, xticklabels=cols.values)
#creating heatmap using seaborn 
plt.show()


# In[53]:


#Above are the variables that are most correlated to SalePrice
#Our assumptions above that OverallQual, GrLiveArea, TotalBsmtArea are correlated with Salesprice are true
#Garage cars and Garage area are strongly correlated but basically the same. We will use garage cars in our analysis
#since the correlation is higher

#Same with TotalBsmtSF and 1stFloorSF. We will keep TotalBsmtSF
#Same with TotRomsAboveGrd and GrLivArea

#onto the scatter plots


# In[16]:


sns.set()
#apply default seaborn scheme
cols = ['SalePrice','OverallQual','GrLivArea','GarageCars','TotalBsmtSF','FullBath','YearBuilt']
#set column names into list
sns.pairplot(df_train[cols],size = 2.5)
#using seaborn to compare the variables using pairplot
#pairplot plots pairwise relationships in a dataset creating a grid of plots with each variable in x and y axis
plt.show()
#shows the plots


# In[55]:


#looks like Basement size = living room area
#prices rise exponentially the newer the house


# In[56]:


#Important questions to ask about Missing Data
#How prevalent is the missing data?
#Is missing data random or does it have a pattern?

#missing data can imply a reduction in sample size which can prevent us from proceeding with the analysis
#need to ensure the missing data process is not biased and hiding an inconvenient truth


# In[35]:


Total = df_train.isnull().sum().sort_values(ascending = False)
#total number of null values for variable data sorted by most missing missing
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending = False)
#setting a percent by dividing sum of missing values
missing_data = pd.concat([Total,percent], axis=1, keys=['Total','Percent'])
missing_data.head(20)


# In[18]:


#the user that is guiding this project feels that variables missing more than >15% of data should be deleted
#i will do that in the next cell
#PoolQC, MiscFeature, Alley will be deleted as they are >15% and these arent feature people look for when buying a home

#GarageX variables are missing the same number of variables. Garage cars will represent this data
#Same logic for BasementX variables

#For electricalm


# In[40]:


df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)
#setting df_train now as a new data frame by dropping the variables where missing values are > 1
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
#setting df_train now as a new data frame by dropping the null variable in the electricity columns


# In[41]:


print(len(df_train.columns))
#checking to see if columns were dropped


# In[ ]:


#lets go on to do analysis on the outliers in our data
#univariate analysis

#we need to establish a threshold that defines an observation as an outlier.
#we will standardize the data - converting data values to have a mean of 0 and standard deviation of 1


# In[43]:


saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis])
#using the standardscaler for the fit_transform function to scale the training data and the scaling parameters
#the model will learn the mean and variance of the features of the training data
#https://stackoverflow.com/questions/29241056/how-does-numpy-newaxis-work-and-when-to-use-it
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
#setting low range of distribution by taking the scaled saleprice
high_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
#setting low range of distribution by taking the scaled saleprice
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)


# In[ ]:


#Data standardization is the process of rescaling the attributes so that they have mean as 0 and variance as 1.
#The ultimate goal to perform standardization is to bring down all the features to a common scale 
#without distorting the differences in the range of the values.
#In sklearn.preprocessing.StandardScaler(), centering and scaling happens independently on each feature.

#observations 
#low range values are similar and not too far from 0
#high range values are further from 0 (last two values are really out of range)

#lets move on to bivariate analysis of sale price vs grlivarea


# In[46]:


var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'],df_train[var]],axis=1)
data.plot.scatter(x=var,y='SalePrice', ylim=(0,800000))


# In[ ]:


#the two at the bottom right do not seem to be following the trend
#we will delete those outliers


# In[45]:


df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]
#sorting df_train by GrLiveArea and keeping the first 2 
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)
#dropping the rows 1299 and 524


# In[ ]:


#bivariate analysis of salesprice vs totalbsmntsf


# In[47]:


var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'],df_train[var]],axis=1)
data.plot.scatter(x=var,y='SalePrice', ylim=(0,800000))


# In[ ]:


#looks ok, no need to delete any data

#lets look for normality and homoscedascity 
#histogram to see kurtosis and skewness
#normal probability plot - to see if data distribution follows the diagnal that represents normal distribution


# In[50]:


sns.distplot(df_train['SalePrice'],fit=norm)
#using seaborn to plot histogram by normal distribution
fig = plt.figure()
#create figure object
res = stats.probplot(df_train['SalePrice'],plot = plt)
#use scipy to create prob plot of saleprice


# In[ ]:


#sale price is not normal distributed
#it shows peakedness, positive skewness does not follow the diagnal line
#lets tranform this data using log transformations


# In[51]:


df_train['SalePrice'] = np.log(df_train['SalePrice'])


# In[52]:


#re-running histogram and probability plot

sns.distplot(df_train['SalePrice'],fit=norm)
#using seaborn to plot histogram by normal distribution
fig = plt.figure()
#create figure object
res = stats.probplot(df_train['SalePrice'],plot = plt)
#use scipy to create prob plot of saleprice


# In[ ]:


#looks good, now lets check out GrLivArea


# In[53]:


sns.distplot(df_train['GrLivArea'],fit=norm)
#using seaborn to plot histogram by normal distribution
fig = plt.figure()
#create figure object
res = stats.probplot(df_train['GrLivArea'],plot = plt)
#use scipy to create prob plot of saleprice


# In[54]:


#there is skewness, lets transform this data
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])

sns.distplot(df_train['GrLivArea'],fit=norm)
#using seaborn to plot histogram by normal distribution
fig = plt.figure()
#create figure object
res = stats.probplot(df_train['GrLivArea'],plot = plt)
#use scipy to create prob plot of saleprice


# In[55]:


#much better! lets try TotalBsmtSF

sns.distplot(df_train['TotalBsmtSF'],fit=norm)
#using seaborn to plot histogram by normal distribution
fig = plt.figure()
#create figure object
res = stats.probplot(df_train['TotalBsmtSF'],plot = plt)
#use scipy to create prob plot of salepric


# In[ ]:


#there is skewnwess
#lots of houses with no basements (value = 0)
#we cannot do log transformation on zero values

#to conduct a log transformation, we will create a variable that can create the effect of hacing or not having
#a basement. then we will do a log transformation to all the non-zero observations and then ignore those with
#value 0

#a log transform of data set containing zero values can be easily handled by numpy.log1p()


# In[57]:


#create column for new variable (one is enough because it's a binary categorical feature)
#if area>0 it gets 1, for area==0 it gets 0
df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
#create new column "HasBsmt"
df_train['HasBsmt'] = 0 
#set all values to 0
df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1
#goes thrrough each index of TotalBsmtSF and sets the value for HasBsmt to 1 if value > 0


# In[58]:


#transform the data
df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])
#find indexes where there HasBsmt = 1 and conduct a log transformation


# In[60]:


sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'],fit=norm)
#using seaborn to plot histogram by normal distribution where totalbsmtsf > 0
fig = plt.figure()
#create figure object
res = stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'],plot = plt)
#use scipy to create prob plot of saleprice


# In[ ]:


#lets search for homoscedascity


# In[62]:


plt.scatter(df_train['GrLivArea'],df_train['SalePrice'])


# In[ ]:


#older versions of this plot had a conic shape - it is normalized
#now lets check saleprice with totalbsmtsf


# In[63]:


plt.scatter(df_train['TotalBsmtSF'],df_train['SalePrice'])


# In[ ]:


#equal level of variance of salesprice across range of totalbsmtsf

#now dummy variables


# In[64]:


df_train = pd.get_dummies(df_train)


# In[ ]:


#comments


# In[ ]:


"Understanding the problem" phase is great, “developing sixth sense” is a practice that is often disregarded.
Guessing which columns to drop, in case of correlating features ideally should be performed based on some 
experiments, like their effect on the model accuracy or something of that ilk.
It’s always better to retain the data than to dismiss it. The threshold of 15% for dropping appears to me 
somewhat arbitrary. As a rule of thumb, statisticians dismiss some feature if it has more than 80 percent 
missing data and doesn’t seem to be significantly important, because at that point the bias from imputation 
is very likely to be more problematic than dropping the whole feature. After that, ideally, it needs a little
bit of experimentation to decide which features to impute and which features to drop. “Best statisticians 
make the fewest assumptions”.
Making decisions based on univariate and scatter plots seems a little dangerous because they can have totally
different relationships on higher dimensions.

