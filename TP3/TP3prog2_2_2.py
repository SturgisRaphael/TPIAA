import numpy as np
from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.linear_model import Lasso
import random

def change_reg_in_liste(reg):
    w_LinearRegrassion = []
    w_LinearRegrassion.append(reg.intercept_)
    for i in reg.coef_:
        w_LinearRegrassion.append(i)
    return  w_LinearRegrassion

diabetes = datasets.load_diabetes()
data = diabetes.data
target = diabetes.target
data_train,data_test,target_train,target_test=train_test_split(data,target,test_size=0.3,random_state=random.seed())

alphas = np.logspace(start=-4, stop=-1, num=100, base=10)
gscv = GridSearchCV(Ridge(), dict(alpha=alphas), cv=10).fit(data_train, target_train)
alpha_r = gscv.best_params_['alpha']
print("alpha_r = ", alpha_r)
gscv = GridSearchCV(Lasso(), dict(alpha=alphas), cv=10).fit(data_train, target_train)
alpha_l = gscv.best_params_['alpha']
print("alpha_l = ", alpha_l)

clf_ridge = Ridge(alpha=alpha_r)
clf_ridge.fit(data_train,target_train)
print("Ridge :\n",clf_ridge.intercept_, clf_ridge.coef_)
Y_pred_rigde = clf_ridge.predict(data_test)

clf_lasso = Lasso(alpha=alpha_l)
clf_lasso.fit(data_train,target_train)
print("\nLasso :\n",clf_lasso.intercept_, clf_lasso.coef_)
Y_pred_lasso = clf_lasso.predict(data_test)

print("\nRigde mean_squared_error : ", mean_squared_error(target_test, Y_pred_rigde))
print("\nLasso mean_squared_error : ", mean_squared_error(target_test, Y_pred_lasso))
