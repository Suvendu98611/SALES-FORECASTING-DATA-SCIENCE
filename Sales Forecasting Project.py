#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6


# In[17]:


display (os.getcwd())


# In[18]:


os.chdir ('C:\\Users\\Suvendu\\OneDrive\\Desktop\\PRJ SALES FORECASTING')


# In[19]:


display (os.getcwd())


# In[20]:


dt = pd.read_csv('Train.csv')
display (dt.head())


# In[21]:


print (dt.shape)


# In[22]:


display (dt.columns)


# In[23]:


display (dt.describe())


# In[24]:


display (dt.info())


# In[25]:


display (dt.apply(lambda x: len(x.unique())))


# In[26]:


display (dt.isnull().sum())


# In[27]:


cat_col = []
for x in dt.dtypes.index:
    if dt.dtypes[x] == 'object':
        cat_col.append(x)
display (cat_col)


# In[28]:


cat_col.remove('Item_Identifier')
cat_col.remove('Outlet_Identifier')
display (cat_col)


# In[29]:


for col in cat_col:
    print(col , len(dt[col].unique()))


# In[30]:


for col in cat_col:
    print(col)
    print(dt[col].value_counts())
    print()
    print ('*' *50)


# In[31]:


miss_bool = dt['Item_Weight'].isnull()
display (miss_bool)


# In[32]:


dt['Item_Weight'].isnull().head(20)


# In[33]:


dt[dt['Item_Weight'].isnull()]


# In[34]:


Item_Weight_null = dt[dt['Item_Weight'].isna()]
display (Item_Weight_null)


# In[35]:


Item_Weight_null['Item_Identifier'].value_counts()


# In[36]:


item_weight_mean = dt.pivot_table(values = "Item_Weight", index = 'Item_Identifier')
display (item_weight_mean)


# In[37]:


display (dt['Item_Identifier'])


# In[38]:


for i, item in enumerate(dt['Item_Identifier']):
    if miss_bool[i]:
        if item in item_weight_mean.index:
            dt['Item_Weight'][i] = item_weight_mean.loc[item]['Item_Weight']
        else:
            dt['Item_Weight'][i] = np.mean(dt['Item_Weight'])


# In[39]:


miss_bool


# In[40]:


result = dt['Item_Weight'].isnull().sum()
display (result)


# In[41]:


result = dt.groupby('Outlet_Size').agg({'Outlet_Size': np.size})
display (result)


# In[42]:


result= dt['Outlet_Size'].isnull().sum()
display (result)


# In[43]:


Outlet_Size_null= dt[dt['Outlet_Size'].isna()]
display (Outlet_Size_null)


# In[44]:


result = Outlet_Size_null['Outlet_Type'].value_counts()
display (result)


# In[45]:


result= dt.groupby (['Outlet_Type','Outlet_Size'] ).agg({'Outlet_Type':[np.size]})
display (result)


# In[46]:


outlet_size_mode = dt.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))
display (outlet_size_mode)


# In[47]:


miss_bool = dt['Outlet_Size'].isnull()
dt.loc[miss_bool, 'Outlet_Size'] = dt.loc[miss_bool, 'Outlet_Type'].apply(lambda x: outlet_size_mode[x])


# In[48]:


display (dt['Outlet_Size'].isnull().sum())


# In[49]:


result = dt.groupby (['Outlet_Type','Outlet_Size'] ).agg({'Outlet_Type':["size"]})
display (result)


# In[50]:


display (sum(dt['Item_Visibility']==0))


# In[51]:


dt.loc[:, 'Item_Visibility'].replace([0], [dt['Item_Visibility'].mean()], inplace=True)


# In[52]:


display(sum(dt['Item_Visibility']==0))


# In[53]:


result = dt['Item_Fat_Content'].value_counts()
display (result)


# In[54]:


dt['Item_Fat_Content'] = dt['Item_Fat_Content'].replace({'LF':'Low Fat', 'reg':'Regular', 'low fat':'Low Fat'})
result = dt['Item_Fat_Content'].value_counts()
display (result)


# In[55]:


dt['New_Item_Type'] = dt['Item_Identifier'].apply(lambda x: x[:2])
display (dt['New_Item_Type'])


# In[56]:


display (dt['New_Item_Type'].value_counts())


# In[57]:


dt['New_Item_Type'] = dt['New_Item_Type'].map({'FD':'Food', 'NC':'Non-Consumable', 'DR':'Drinks'})
display (dt['New_Item_Type'].value_counts())


# In[58]:


display (dt['Item_Fat_Content'].value_counts())


# In[59]:


result = dt.groupby (['New_Item_Type','Item_Fat_Content'] ).agg({'Outlet_Type':[np.size]})
display (result)


# In[60]:


dt.loc[dt['New_Item_Type']=='Non-Consumable', 'Item_Fat_Content'] = 'Non-Edible'
result =  (dt['Item_Fat_Content'].value_counts())
display (result)


# In[61]:


result = dt.groupby (['New_Item_Type','Item_Fat_Content'] ).agg({'Outlet_Type':[np.size]})
display (result)


# In[62]:


dt['Outlet_Years'] = 2024 - dt['Outlet_Establishment_Year']
print (dt['Outlet_Years'])


# In[63]:


display (dt.head())


# In[64]:


sns.distplot(dt['Item_Weight'])
plt.show()


# In[65]:


sns.distplot(dt['Item_Visibility'])
plt.show()


# In[66]:


sns.distplot(dt['Item_MRP'])
plt.show()


# In[67]:


sns.distplot(dt['Item_Outlet_Sales'])
plt.show()


# In[68]:


dt['Item_Outlet_Sales'] = np.log(1+dt['Item_Outlet_Sales'])
display (dt['Item_Outlet_Sales'])


# In[69]:


sns.distplot(dt['Item_Outlet_Sales'])
plt.show()


# In[70]:


sns.countplot(x = dt["Item_Fat_Content"])
plt.show()


# In[71]:


l = list(dt['Item_Type'].unique()) 
chart = sns.countplot(x =dt["Item_Type"])
chart.set_xticklabels(labels=l, rotation=90)
plt.show()


# In[72]:


sns.countplot(x= dt['Outlet_Establishment_Year'])
plt.show()


# In[73]:


sns.countplot(x=dt['Outlet_Size'])
plt.show()


# In[74]:


sns.countplot(x=dt['Outlet_Location_Type'])
plt.show()


# In[75]:


sns.countplot(x= dt['Outlet_Type'])
plt.show()


# In[76]:


display(dt.head(3))


# In[77]:


dtc= dt.iloc[:,[1,3,5,7,11,13]]
display (dtc)


# In[78]:


corr = dtc.corr()
display (corr)


# In[79]:


sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()


# In[80]:


display (dt.head())


# In[81]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dt['Outlet'] = le.fit_transform(dt['Outlet_Identifier'])
display (dt['Outlet'])


# In[82]:


cat_col = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'New_Item_Type']
for col in cat_col:
    dt[col] = le.fit_transform(dt[col])
display (dt.head())   


# In[83]:


dt = pd.get_dummies(dt, columns=['Item_Fat_Content', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'New_Item_Type'],dtype = int )
display (dt.head())


# In[84]:


X = dt.drop(columns=['Outlet_Establishment_Year', 'Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales'])
display (X.head())


# In[85]:


y = dt['Item_Outlet_Sales']
display (y.head())


# In[ ]:





# In[86]:


from sklearn import metrics 
display (",   ".join(metrics.get_scorer_names()))


# In[87]:


from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
def train(model, X, y):

    print ("Train Test Split")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    print (X.shape, y.shape)
    print (X_train.shape, X_test.shape ,  y_train.shape, y_test.shape)
    
    # training the model
    model.fit(X_train, y_train)       
   
    # perform cross-validation
    cv_score = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)
    print("Model Report")
    print ('Scoring - neg_mean_squared_error')
    print ( cv_score )    
    cv_score = np.abs(np.mean(cv_score))    
    print ('ABS Average of - neg_mean_squared_error',cv_score )       
    cv_score = cross_val_score(model, X, y,  cv=5)
    print ()
    print ('R2 Score ')
    print ( cv_score )    
    cv_score = np.mean(cv_score)     
    print ('Average R2 Score ',cv_score)    
    print ()
    
    # Display Accuracy
    print ('Accuracy')
    print ('Accuracy of Test data')
    y_test_pred = model.predict(X_test)
    print('R2_Score:', r2_score(y_test,y_test_pred))
    print ('Accuracy of Training data')
    y_train_pred = model.predict(X_train)
    print('R2_Score:', r2_score(y_train,y_train_pred))
    print ('Accuracy of Complete data')
    y_pred = model.predict(X)
    print('R2_Score:', r2_score(y,y_pred))
    print ()

    # Display graph with actual and predicted values 
    
    plt.subplot (212)
    print ('Display actual and predicted values')
    sns.regplot( x =y, y= y_pred, scatter_kws={"color": "b"}, 
            line_kws={"color": "r"},ci = None)
    plt.show()


# In[88]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
train(model, X, y)
coef = pd.Series(model.coef_, X.columns).sort_values()
print (coef)
coef.plot(kind='bar', title="Model Coefficients")
plt.show()


# In[89]:


from sklearn.linear_model import Ridge
model = Ridge()
train(model, X,y)
coef = pd.Series(model.coef_, X.columns).sort_values()
coef.plot(kind='bar', title="Model Coefficients")
plt.show()


# In[90]:


from sklearn.linear_model import Lasso
model = Lasso()
train(model, X,y)
coef = pd.Series(model.coef_, X.columns).sort_values()
coef.plot(kind='bar', title="Model Coefficients")
plt.show()


# In[91]:



from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
train(model, X,y)
coef = pd.Series(model.feature_importances_, X.columns).sort_values(ascending=False)
coef.plot(kind='bar', title="Feature Importance")
plt.show()


# In[92]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
train(model, X,y)
coef = pd.Series(model.feature_importances_, X.columns).sort_values(ascending=False)
coef.plot(kind='bar', title="Feature Importance")
plt.show()


# In[93]:


from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
train(model, X,y)
coef = pd.Series(model.feature_importances_, X.columns).sort_values(ascending=False)
coef.plot(kind='bar', title="Feature Importance")
plt.show()


# In[94]:


from lightgbm import LGBMRegressor
model = LGBMRegressor()
train(model, X,y)
coef = pd.Series(model.feature_importances_, X.columns).sort_values(ascending=False)
coef.plot(kind='bar', title="Feature Importance")
plt.show()


# In[95]:


from xgboost import XGBRegressor
model = XGBRegressor()
train(model, X,y)
coef = pd.Series(model.feature_importances_, X.columns).sort_values(ascending=False)
coef.plot(kind='bar', title="Feature Importance")
plt.show()


# In[96]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print (X.shape, y.shape)
print (X_train.shape, X_test.shape ,  y_train.shape, y_test.shape)


# In[98]:


max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10]


# In[99]:


random_grid = {
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)


# In[100]:


from sklearn.model_selection import RandomizedSearchCV
rf = RandomForestRegressor()
rf=RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = -1)
display (rf.fit(X_train, y_train))


# In[101]:


print(rf.best_params_)
print(rf.best_score_)

# Display Accuracy
print ()
print ('Accuracy')
print ('Accuracy of Test data')
y_test_pred = rf.predict(X_test)
print('R2_Score:', r2_score(y_test,y_test_pred))
print ('Accuracy of Training data')
y_train_pred = rf.predict(X_train)
print('R2_Score:', r2_score(y_train,y_train_pred))
print ('Accuracy of Complete data')
y_pred = rf.predict(X)
print('R2_Score:', r2_score(y,y_pred))
print ()


  


# In[102]:


# Display graph with actual and predicted values 
    
plt.subplot (212)
print ('Display actual and predicted values')
sns.regplot( x =y, y= y_pred, scatter_kws={"color": "b"}, line_kws={"color": "r"},ci = None)
plt.show()


# In[103]:


from scipy.stats import uniform, randint
params = {
    "gamma": uniform(0, 0.5),
    "learning_rate": uniform(0.03, 0.3), # default 0.1 
    "max_depth": randint(2, 6), # default 3
    "n_estimators": randint(100, 150), # default 100
    "subsample": uniform(0.6, 0.4)
}


# In[104]:


lgb=LGBMRegressor()
lgb = RandomizedSearchCV(estimator = lgb, param_distributions = params,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)
lgb.fit(X,y)


# In[105]:


print(lgb.best_params_)
print(lgb.best_score_)

# Display Accuracy
print ()
print ('Accuracy')
print ('Accuracy of Test data')
y_test_pred = lgb.predict(X_test)
print('R2_Score:', r2_score(y_test,y_test_pred))
print ('Accuracy of Training data')
y_train_pred = lgb.predict(X_train)
print('R2_Score:', r2_score(y_train,y_train_pred))
print ('Accuracy of Complete data')
y_pred = lgb.predict(X)
print('R2_Score:', r2_score(y,y_pred))
print ()

# Display graph with actual and predicted values 
    
plt.subplot (212)
print ('Display actual and predicted values')
sns.regplot( x =y, y= y_pred, scatter_kws={"color": "b"}, line_kws={"color": "r"},ci = None)
plt.show()


# In[106]:


params = {
    "gamma": uniform(0, 0.5),
    "learning_rate": uniform(0.03, 0.3), # default 0.1 
    "max_depth": randint(2, 6), # default 3
    "n_estimators": randint(100, 150), # default 100
    "subsample": uniform(0.6, 0.4)
}


# In[107]:


xgb = RandomizedSearchCV(estimator = model, param_distributions = params,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)
xgb.fit(X,y)


# In[108]:


print(xgb.best_params_)
print(xgb.best_score_)

# Display Accuracy
print ()
print ('Accuracy')
print ('Accuracy of Test data')
y_test_pred = xgb.predict(X_test)
print('R2_Score:', r2_score(y_test,y_test_pred))
print ('Accuracy of Training data')
y_train_pred = xgb.predict(X_train)
print('R2_Score:', r2_score(y_train,y_train_pred))
print ('Accuracy of Complete data')
y_pred = xgb.predict(X)
print('R2_Score:', r2_score(y,y_pred))
print ()

# Display graph with actual and predicted values 
    
plt.subplot (212)
print ('Display actual and predicted values')
sns.regplot( x =y, y= y_pred, scatter_kws={"color": "b"}, line_kws={"color": "r"},ci = None)
plt.show()



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




