import time
import sys
! pip install numpy
import numpy as np
! pip install matplotlib
import matplotlib.pyplot as plt
import math
import os
from matplotlib.pyplot import imread
! pip install patchify
from patchify import patchify

np.random.seed(1)




#load clustering data
X = np.load("./data_clustering.npy")
plt.scatter(X[:,0], X[:,1])
plt.show()




#initialise parameters for the EM algorithm
def initialise_parameters(m, X):
    C = X[np.random.choice(X.shape[0], m)]
    return C

C = initialise_parameters(4, X)
print(C)





#E step of the EM algo
def E_step(C, X):
    # YOUR CODE HERE
    arrayL = np.empty((len(X),2), float)
    itr = -1
    for i in X:
        itr += 1
        centroid = C[0]
        distance = (X[itr][0]-C[0][0])**2 + (X[itr][1]-C[0][1])**2
        for j in range(1,len(C)):
            dist1 = (X[itr][0]-C[j][0])**2 + (X[itr][1]-C[j][1])**2
            if dist1 < distance:
                centroid = C[j]
                distance = dist1
        #print(centroid)
        arrayL[itr] = centroid
        #print(X[itr], L[itr])

    #print(arrayL)
                
    return arrayL
    
L = E_step(C, X)
plt.scatter(L[:, 0], L[:, 1])
plt.show()




#M step of the EM algo
def M_step(C, X, L):
    # YOUR CODE HERE
    newC = np.empty((len(C), 2),float)
    for i in range (len(C)):
        xsum = 0
        ysum = 0
        count = 0
        for j in range(len(X)):
            if (L[j][0] == C[i][0] and L[j][1] == C[i][1]):
                xsum += X[j][0]
                ysum += X[j][1]
                count += 1
        if count != 0:
            xmean = xsum / count
            ymean = ysum / count
            newC[i][0] = xmean
            newC[i][1] = ymean
    return newC
    pass

print('Before:')
print(C)
print('\nAfter:')
new_C = M_step(C, X, L)
print(new_C)




#implement K-means classification
def kmeans(X, m, threshold):
    # YOUR CODE HERE
        
    #initialisation
    C = initialise_parameters(m, X)
    
    #1 iteration
    L= E_step(C,X)
    C = M_step(C, X, L)
    loss = 0
    for j in range (len(X)):
        loss += ((X[j][0] - L[j][0])**2+(X[j][1] - L[j][1])**2)
    MSE1 = loss / len(X)
    
    #2 iterations
    L= E_step(C,X)
    C = M_step(C, X, L)
    loss = 0
    for j in range (len(X)):
        loss += ((X[j][0] - L[j][0])**2+(X[j][1] - L[j][1])**2)
    MSE2 = loss / len(X)
    
    #more iterations
    while ((MSE1 - MSE2) >= threshold):
        MSE1 = MSE2
    
        L = E_step(C, X)
        C = M_step(C, X, L)
        
        loss = 0
        for j in range (len(X)):
            loss += ((X[j][0] - L[j][0])**2+(X[j][1] - L[j][1])**2)
        MSE2 = loss / len(X)
        diff = MSE2 - MSE1
    return C, L
    pass

#CODE TO DISPLAY YOUR RESULTS. DO NOT MODIFY.
C_final, L_final = kmeans(X, 4, 1e-6)
print('Initial Parameters:')
print(C)
print('\nFinal Parameters:')
print(C_final)

def allocator(X, L, c):
    cluster = []
    for i in range(L.shape[0]):
        if np.array_equal(L[i, :], c):
            cluster.append(X[i, :])
    return np.asarray(cluster)

colours = ['r', 'g', 'b', 'y']
for i in range(4):
    cluster = allocator(X, L_final, C_final[i, :])
    plt.scatter(cluster[:,0], cluster[:,1], c=colours[i])

plt.show()




#gradient descent
x_train, _, y_train, _ = np.load("./data_regression.npy")
plt.plot(x_train,y_train,'o')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Training data")
plt.ylim([-1,3])
plt.show()




#get polynomial features
def get_polynomial_features(x,degree=5):
    # YOUR CODE HERE
    FeatureArray = np.empty((len(x), degree + 1), float)
    for i in range (len(x)):
        for j in range (degree + 1):
            FeatureArray[i][j] = x[i]**j
    return FeatureArray
    
    pass


# get polynomial features
X_train = get_polynomial_features(x_train,degree=2)




#initialise theta
def initialise_parameters(n):
    # YOUR CODE HERE
    theta = []
    for i in range (n):
        theta.append(np.random.uniform(-10,10))
    return theta
    
    pass
    
    
# initialize theta
theta = initialise_parameters(X_train.shape[1])
print(theta)




#mean squared error
def ms_error(X, theta, y):
    # YOUR CODE HERE
    
    return np.transpose(y - X @ theta) @ (y - X @ theta) / len(y)
    pass

print(ms_error(X_train, theta, y_train))




#implement gradient descent
def grad(X, theta, Y):
    # YOUR CODE HERE
    #gradient of dL/d(theta) can be mathematically reduced to the following equation
    gradient = (-2 * np.transpose(Y) @ X + 2 * np.transpose(theta) @ np.transpose(X) @ X) / len(Y)
    return gradient
    
    
    pass

print(grad(X_train, theta, y_train))




#get batch descent
def batch_descent(X, Y, iterations, learning_rate):
    # YOUR CODE HERE
    
    theta = initialise_parameters(X.shape[1])
    L = []
    for i in range (iterations):
        theta = theta - 0.5 * grad(X, theta, Y)
        L.append(ms_error(X, theta, Y))
    return theta, L
    pass

    
#REPORTING CODE. YOU MAY NEED TO MODIFY THE LEARNING RATE OR NUMBER OF ITERATIONS
new_theta, L = batch_descent(X_train, y_train, 5000, 0.5)
plt.plot(L)
plt.title('Mean Squared Error vs Iterations')
plt.show()
print('New Theta: \n', new_theta)
print('\nFinal Mean Squared Error: \n', ms_error(X_train, new_theta, y_train))

def get_prediction(X,theta):
    pred = X@theta
    return pred

x_fit = np.linspace(-0.7, 0.8, 1000)
X_fit = get_polynomial_features(x_fit,degree=2)
pred_y_train = get_prediction(X_fit,new_theta)

# plot results
plt.plot(x_train,y_train,'o',label='data point')
plt.plot(x_fit,pred_y_train,label='fitting result')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('show fitting result')
plt.ylim([-1,3])
plt.show()
