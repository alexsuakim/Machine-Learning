import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #This is for 3d scatter plots.
import math
import random
import functools
import scipy.stats




#show data
X = np.load("./data.npy")
fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], X[:, 2])
plt.show()




#initialise the parameters X and K
def initialise_parameters(X, K):
    # YOUR CODE HERE
    m, n = X.shape
    
    #sigma
    arrays = []
    for i in range (K):
        value = np.random.choice(X.ravel(), (n,n))
        mean = np.random.choice(X.ravel(), (n,n))
        diff = value - mean
        diff_squared = diff * diff.T
        matrix = np.diag(np.diag(diff_squared))
        arrays.append(matrix)
    sigma = np.stack(arrays)
    
    
    #mu
    mu = np.random.choice(X.ravel(), (K,n))
    
    #pi
    pi = np.ones(K) / K
    
    return sigma, mu, pi
    
K = 4
sigma, mu, pi = initialise_parameters(X[:, :3], K)
print('\nSigma: \n', sigma)
print('\nMu: \n', mu)
print('\nPi: \n', pi)




#implement the E step of GMM
def E_step(pi, mu, sigma, X):
    # YOUR CODE HERE
    m,n = X.shape
    K = len(pi)
    r = np.empty((m, K))
    for i in range (m):
        numerator = np.zeros(K)
        for k in range (K):
            weight = pi[k]
            gaussian = scipy.stats.multivariate_normal.pdf(X[i], mu[k], sigma[k], allow_singular=True)
            numerator[k] = weight * gaussian
        denominator = np.sum(numerator)
        r[i,:] = numerator / denominator
            
    return r

responsibilities = E_step(pi, mu, sigma, X[:, :3])
print(responsibilities)




#implement the M step of GMM
def M_step(r, X):
    # YOUR CODE HERE
    
    #shape
    m,n = X.shape
    k = len(r[0])
    
    #mu
    mu = np.dot(r.T, X) / np.sum(r, axis = 0).reshape(k,1)
    #mu = np.zeros((k,n))
    #for i in range (k):
        #sum = 0
        #for j in range (m):
            #sum += r[j,i] * X[j]
        #mu[i] = sum / np.sum(r[:,i])
    
    #sigma    
    sigma = np.zeros((k,n,n))
    for i in range (k):
        sum = 0
        for j in range (m):
            sum += r[j,i] * np.outer((X[j] - mu[i]), (X[j] - mu[i]).T) 
        sigma[i] = sum / np.sum(r[:,i])
    
    
    #pi
    pi = np.sum(r,axis=0) / np.sum(r)
    #pi = np.zeros(k)
    #for i in range (k):
        #pi[i] = np.sum(r[:,i]) /  np.sum(r)
    
    return mu, sigma, pi

mu, sigma, pi = M_step(responsibilities, X[:, :3])
print('\nSigma: \n', sigma)
print('\nMu: \n', mu)
print('\nPi: \n', pi)




#classification
def classify(pi, mu, sigma, x):
    # YOUR CODE HERE
    
    #size
    kmax = len(pi)
    
    #maximise
    maxarray = np.zeros((kmax))
    for i in range (kmax):
        maxarray[i] = scipy.stats.multivariate_normal.pdf(x, mu[i], sigma[i], allow_singular=True) * pi[i]
    k = np.argmax(maxarray)
    return k

print(classify(pi, mu, sigma, X[270, :3]))




#implement the EM algorithm
def EM(X, K, iterations):
    # YOUR CODE HERE
    
    #initialise
    sigma, mu, pi = initialise_parameters(X,K)
    
    #iterate E-M
    for i in range(iterations):
        r = E_step(pi,mu,sigma,X)
        mu, sigma, pi = M_step(r,X)
        
    return mu, sigma, pi


#Test code. Leave it aloooooone!
iterations = 30
K = 3
mu_1, sigma_1, pi_1 = EM(X[:, :3], K, iterations)
print('\nSigma: \n', sigma_1)
print('\nMu: \n', mu_1)
print('\nPi: \n', pi_1)

def allocator(pi, mu, sigma, X, k):
    N = X.shape[0]
    cluster = []
    for ix in range(N):
        prospective_k = classify(pi, mu, sigma, X[ix, :])
        if prospective_k == k:
            cluster.append(X[ix, :])
    return np.asarray(cluster)

colours = ['r', 'g', 'b']
fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111, projection='3d')
for k in range(K):
    cluster = allocator(pi_1, mu_1, sigma_1, X[:, :3], k)
    ax.scatter(cluster[:,0], cluster[:,1], cluster[:, 2], c=colours[k])
plt.show()




#image segmentation
image = plt.imread('mandm.png')
plt.imshow(image)
plt.show()

def image_segmentation(image, K, iterations):
    #reshape
    m,n,o = image.shape
    img = image.reshape(m*n, o)
    
    #initialise
    sigma, mu, pi = initialise_parameters(img,K)
    
    #iterate E-M
    for i in range (iterations):
        r = E_step(pi, mu, sigma, img)
        mu, sigma, pi = M_step(r,img)
    
    #classify
    cluster = np.zeros((m*n), int)
    for i in range (m*n):
        cluster[i] = classify(pi, mu, sigma, img[i,:])
    cluster = cluster.reshape(m,n)
    
    return cluster

# test code, leave it alone!
import time
start = time.time()
gmm_labels = image_segmentation(image, 5, 10)
end = time.time()
print(f'It takes {end-start} seconds to segement the image.')
colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255], [0, 0, 0]]
segemented_image = np.zeros_like(image, dtype=np.int32)
m, n, _ = segemented_image.shape
for i in range(m):
    for j in range(n):
        segemented_image[i, j] = np.array(colors[gmm_labels[i, j]])
plt.imshow(segemented_image)
plt.show()
