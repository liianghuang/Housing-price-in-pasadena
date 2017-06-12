
# coding: utf-8

# In[595]:

##coding test from Li-Iang Huang
## Basic Data Analysis(Machine learning technique on Redfin data)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.cross_validation import train_test_split
from scipy.stats import skew
import seaborn as sns


# In[652]:

#read data
pasadena = pd.read_csv('pasadena.csv')

#check missing data
pasadena.isnull().any()
pasadena = pasadena.fillna(method='ffill')
pasadena = pasadena.dropna(axis=1,how='all')

#split data into training and test set
train, test = train_test_split(pasadena, test_size = 0.2)


# In[653]:

print(pasadena.columns)
#relation between number of bedrooms and pricep
plt.scatter(pasadena['BEDS'],pasadena['PRICE'])
plt.ylabel('Price')
plt.xlabel('Number of beds')
plt.title('BEDS vs PRICE')
plt.savefig('BEDS.png')
plt.show()
#relation between SQUARE FEET and price
plt.scatter(pasadena['$/SQUARE FEET'],pasadena['PRICE'])
plt.ylabel('Price')
plt.xlabel('SQUARE FEET')
plt.title('SQUARE FEET vs PRICE')
plt.savefig('SQUARE_FEET.png')
plt.show()
#relation between nYEAR BUILT to price
plt.scatter(pasadena['YEAR BUILT'],pasadena['PRICE'])
plt.ylabel('Price')
plt.xlabel('YEAR BUILT')
plt.title('YEAR BUILT vs PRICE')
plt.savefig('YEAR_BUILT.png')
plt.show()
#relation between ZIP to price
plt.scatter(pasadena['ZIP'],pasadena['PRICE'])
plt.ylabel('Price')
plt.xlabel('ZIP')
plt.title('ZIP vs PRICE')
plt.savefig('ZIP.png')
plt.show()


# In[654]:

#box plot FAVORITE/price
var = 'PROPERTY TYPE'
data = pd.concat([pasadena['PRICE'], pasadena[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="PRICE", data=data)
sns.plt.savefig('property.png')
sns.plt.show()


# In[655]:

#show the distribution of house prices
prices = pd.DataFrame({"price":train['PRICE'], "log(price + 1)":np.log1p(train["PRICE"])})
prices.hist()
plt.savefig('dist_price.png')
plt.show()


# In[624]:

#log transformation prices
train["PRICE"] = np.log1p(train["PRICE"])
print(train["PRICE"])
#log transform numeric features
numeric_feats = pasadena.dtypes[pasadena.dtypes != "object"].index

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

pasadena[skewed_feats] = np.log1p(pasadena[skewed_feats])

#use only numerical data
pasadena_nu = pasadena[numeric_feats]

# #removing outliers
# def outlier_removing(df):
#     low = .05
#     high = .95
#     quant_df = df.quantile([low, high])

#     df = df.apply(lambda x: x[(x>quant_df.loc[low,x.name]) & 
#                                     (x < quant_df.loc[high,x.name])], axis=0)

#     df.dropna(inplace=True)
#     return(df)
# outlier_removing(pasadena)

# ##plot the data again
# #relation between number of bedrooms and pricep
# plt.scatter(pasadena['BEDS'],pasadena['PRICE'])
# plt.show()
# #relation between SQUARE FEET and price
# plt.scatter(pasadena['$/SQUARE FEET'],pasadena['PRICE'])
# plt.show()
# #relation between nYEAR BUILT to price
# plt.scatter(pasadena['YEAR BUILT'],pasadena['PRICE'])
# plt.show()
# #relation between ZIP to price
# plt.scatter(pasadena['ZIP'],pasadena['PRICE'])
# plt.show()



#use catogorical data
pasadena_prop = pd.get_dummies(pasadena['PROPERTY TYPE'])
pasadena = pd.concat([pasadena_prop,pasadena_nu], axis=1)


# In[625]:

#see if there are anything wrong with data
pasadena


# In[626]:

pasadena = pasadena.fillna(pasadena.mean())


# In[627]:

#creating matrices for sklearn
X_train = pasadena[:train.shape[0]]
X_test = pasadena[train.shape[0]:]
y = train.PRICE
X_train.info()
y.isnull()


# In[628]:

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score


# In[629]:

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)


# In[630]:

model_ridge = Ridge()


# In[660]:

alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75, 80, 90]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 
            for alpha in alphas]


# In[661]:

cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation")
plt.xlabel("alpha")
plt.ylabel("rmse")
plt.show()


# In[633]:

cv_ridge.min()


# In[634]:

#Lasso model
model_lasso = LassoCV(alphas = [0.2, 0.1, 0.001, 0.0005, 0.00001]).fit(X_train, y)


# In[635]:

rmse_cv(model_lasso).mean()


# In[636]:

coef = pd.Series(model_lasso.coef_, index = X_train.columns)


# In[637]:

print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")


# In[638]:

imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])


# In[639]:

imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")
plt.savefig('coeff_lasso.png')
plt.show()


# In[640]:

#let's look at the residuals as well:
plt.rcParams['figure.figsize'] = (6.0, 6.0)

preds = pd.DataFrame({"preds":model_lasso.predict(X_train), "true":y})
preds["residuals"] = preds["true"] - preds["preds"]
preds.plot(x = "preds", y = "residuals",kind = "scatter")
plt.savefig('residual.png')
plt.show()


# In[641]:

#Now try neural net
from keras.layers import Dense
from keras.models import Sequential
from keras.regularizers import l1
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


# In[642]:

X_train = StandardScaler().fit_transform(X_train)


# In[643]:

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y, random_state = 3)


# In[644]:

X_tr.shape


# In[645]:

X_tr


# In[646]:

model = Sequential()
#model.add(Dense(256, activation="relu", input_dim = X_train.shape[1]))
model.add(Dense(1, input_dim = X_train.shape[1], W_regularizer=l1(0.001)))

model.compile(loss = "mse", optimizer = "adam")


# In[647]:

model.summary()


# In[648]:


hist = model.fit(X_tr, y_tr, validation_data = (X_val, y_val))


# In[670]:

pd.Series(model.predict(X_val)[:,0]).hist()
plt.xlabel('log price')
plt.title('Prediction by neutral net')
plt.savefig('NN_predict.png')
plt.show()
plt.hist(np.log1p(y_val))
plt.xlabel('log price')
plt.title('Test data')
plt.savefig('test_price.png')
plt.show()
print(pd.Series(model.predict(X_val)[:,0]))
print(np.log1p(y_val))


# In[ ]:



