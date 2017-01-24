from sklearn.linear_model import LinearRegression
import numpy as np


data = np.genfromtxt('ex1data1.txt', delimiter=',')
X = data[:, 0]
y = data[:, 1]

X = np.mat(X).transpose()

reg = LinearRegression()
reg.fit(X, y)
print("Slope: ", reg.coef_, "Intercept: ", reg.intercept_)

print("for 3.5, ", reg.predict(3.5)*10000)
print("for 7, ", reg.predict(7)*10000)
