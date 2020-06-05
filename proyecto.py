# -*- coding: utf-8 -*-
"""
Created on Wed May 27 18:34:25 2020

@author: marvi
"""

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

datos = pd.read_csv('covidgt.csv',sep=';')
print(datos)

x = datos['fecha'].values.reshape(-1, 1) # necesitamos un array de 2D para SkLearn
y = datos['total'].values.reshape(-1, 1)
#plt.scatter(x,y)



model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)
'''
plt.scatter(x, y)
plt.plot(x, y_pred, color='r')
plt.show()



rmse = np.sqrt(mean_squared_error(y,y_pred))
r2 = r2_score(y,y_pred)
print ('RMSE: ' + str(rmse))
print ('R2: ' + str(r2))
'''
poly = PolynomialFeatures(degree=3, include_bias=False)
x_poly = poly.fit_transform(x)
#print(x)
#print(x_poly)

model.fit(x_poly, y)
y_pred = model.predict(x_poly)

print(y_pred)

plt.scatter(x, y)
plt.plot(x, y_pred, color='r')
plt.show()
 
rmse = np.sqrt(mean_squared_error(y,y_pred))
r2 = r2_score(y,y_pred)
print ('RMSE: ' + str(rmse))
print ('R2: ' + str(r2))
