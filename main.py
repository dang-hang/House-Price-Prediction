import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv("Housing.csv")
data.dropna(inplace=True)
from sklearn.model_selection import train_test_split
X=data.drop(['median_house_value'],axis=1)
y=data['median_house_value']
#print(X,y)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
train_data=X_train.join(y_train)
#print(train_data.hist(figsize=(15,8)))
#print(train_data)
'''plt.figure(figsize=(15,8))
sns.heatmap(train_data.loc[:, train_data.columns!='ocean_proximity'].corr(), annot=True, cmap="YlGnBu")'''
train_data['total_rooms']=np.log(train_data['total_rooms']+1)
train_data['total_bedrooms']=np.log(train_data['total_bedrooms']+1)
train_data['population']=np.log(train_data['population']+1)
train_data['households']=np.log(train_data['households']+1)
#train_data.hist(figsize=(15,8))
#print(train_data.ocean_proximity.value_counts())
train_data=train_data.join(pd.get_dummies(train_data.ocean_proximity)).drop(['ocean_proximity'],axis=1)
#print(train_data)
train_data['bedroom_ratio']=train_data['total_bedrooms']/train_data['total_rooms']
train_data['household_rooms']=train_data['total_rooms']/train_data['households']

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
X_train,y_train=train_data.drop(['median_house_value'],axis=1),train_data['median_house_value']
X_train_s=scaler.fit_transform(X_train)
reg=LinearRegression()
reg.fit(X_train,y_train)


test_data=X_test.join(y_test)
test_data['total_rooms']=np.log(test_data['total_rooms']+1)
test_data['total_bedrooms']=np.log(test_data['total_bedrooms']+1)
test_data['population']=np.log(test_data['population']+1)
test_data['households']=np.log(test_data['households']+1)
#train_data.hist(figsize=(15,8))
#print(train_data.ocean_proximity.value_counts())
print(test_data)
test_data=test_data.join(pd.get_dummies(test_data.ocean_proximity)).drop(['ocean_proximity'],axis=1)
#print(train_data)
test_data['bedroom_ratio']=test_data['total_bedrooms']/test_data['total_rooms']
test_data['household_rooms']=test_data['total_rooms']/test_data['households']

X_test,y_test=test_data.drop(['median_house_value'],axis=1),test_data['median_house_value']
X_test_s=scaler.transform(X_test)
print(reg.score(X_test_s,y_test))


from sklearn.ensemble import RandomForestRegressor

forest=RandomForestRegressor()
forest.fit(X_train_s,y_train)
forest.score(X_test_s,y_test)


from sklearn.model_selection import GridSearchCV

forest=RandomForestRegressor()
param_grid={
    "n_estimators": [100,200,300],
    "max_feature": [2,4],
    "max_depth":[None,4,8]
}
grid_search=GridSearchCV(forest,param_grid,cv=5,scoring='neg_mean_squared_error',return_train_score=True)
grid_search.fit(X_train_s,y_train)
best_forest=grid_search.best_estimator_
best_forest.score(X_test_s,y_test)