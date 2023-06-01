from sklearn import linear_model
import numpy as np
import pandas as pd
import math

cancerdata = pd.read_csv("cancer.csv")
# in cancerdata  We have to find the correlation  between thetexture_mean and radius_mean
print(cancerdata.head())
X = cancerdata.texture_mean
Y = cancerdata.radius_mean
print(X)

# function to get the values of w and b until reaching the local minmum


def gradient_descent(x, y):
    # initial value of w and b
    current_weight = 0
    current_bias = 0
    # initialize number of steps
    iterations = 30000
    # Number of data points n
    n = len(x)
    # Initialize learning rate
    learning_rate = 0.002551

    previous_loss = 0

    for i in range(iterations):
        y_pred = current_weight * x + current_bias
        loss_function = (1/n) * sum([val**2 for val in (y-y_pred)])
        md = -(2/n)*sum(x*(y-y_pred))
        bd = -(2/n)*sum(y-y_pred)
        current_weight = current_weight - learning_rate * md
        current_bias = current_bias - learning_rate * bd
        if math.isclose(loss_function, previous_loss, rel_tol=1e-20):
            break
        previous_loss = loss_function
        print("w {}, b {}, loss_function {}, iteration {}".format(
            current_weight, current_bias, loss_function, i))


gradient_descent(X, Y)

reg = linear_model.LinearRegression()
# find the values of w and b that best fit the linear regression to compare it with what we git
reg.fit(cancerdata[['texture_mean']], cancerdata.radius_mean)

print(reg.coef_, reg.intercept_)
S