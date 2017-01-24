import numpy as np
import matplotlib.pyplot as plt


def see_plot(X, y):
    plt.scatter(X, y)
    plt.show()


def normalize(feature):
    mu = np.mean(feature)
    f_range = max(feature) - min(feature)
    X = (feature - mu)/f_range
    return X


def computeCost(X, y, theta):
    m = len(y)

    hx = np.dot(X, theta)
    hx = hx.reshape((hx.shape[0]))
    avg = 1/(2*m)
    
    sqrerror = (hx-y)**2
    J = avg*sum(sqrerror)
    
    return J


def gradientDescent(X, y, theta, alpha, epochs):
    J_hist = np.array([0 for i in range(epochs)])
    m = len(y)

    parameters = theta.shape[0]
    tmp = [0 for i in range(parameters)]
    
    for epoch in range(epochs):
        hx = np.dot(X, theta)
        hx = hx.reshape((hx.shape[0]))
        
        err = hx-y

        for i in range(parameters):
            tmp[i] = theta[i] - alpha*(1/m)*sum(err*X[:, i])

        for i in range(parameters):
            theta[i] = tmp[i]
        
##        theta_zero = theta[0] - alpha*(1/m)*sum(err*X[:, 0])
##        theta_one = theta[1] - alpha*(1/m)*sum(err*X[:, 1])

##        theta[0] = theta_zero
##        theta[1] = theta_one

        J = computeCost(X, y, theta)
##        print(">epoch: {0}, alpha: {1}, error: {2}".format(epoch, alpha, J))
        J_hist[epoch] = J

    return theta, J_hist


def make_predictions(theta):
    x = float(input("Enter population: "))
    profit = theta[0] + theta[1]*x
    print("Predicted profit: ", profit*10000)


data = np.genfromtxt('ex1data1.txt', delimiter=',')
X = data[:,0]   # input features
y = data[:,1]   # target
see_plot(X, y)
m = len(y)      # no. of training examples

# X = normalize(X)

x0 = np.ones((m, 1))
X = X.reshape((X.shape[0],1)) 	# matrix X

X = np.concatenate((x0, X), axis=1)
theta = np.zeros((2, 1))

iterations = 1500
alpha = 0.01

J = computeCost(X, y, theta)
print("Initial cost on zero theta values = ", J)

print("Performing Gradient Descent for {} ITERATIONS".format(iterations))
print("Learning Rate: {}".format(alpha))
(theta, J_vals) = gradientDescent(X, y, theta, alpha, iterations)
print("After Gradient descent, found theta to be: ", theta)

while True:
    halt = input("Our Model is ready! Press Q when you want to stop making predictions")
    if halt == 'Q' or halt == 'q':
        break
    make_predictions(theta)
