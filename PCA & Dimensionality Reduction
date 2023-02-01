import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #This is for 3d scatter plots.
import math
import random
import os
import scipy
from matplotlib.pyplot import imread
from PIL import Image
np.random.seed(13579201)





#show image exampe
width = 64
height = 128
dimension = (height, width, 3)
images = []
filename = []
for file in os.listdir("./train"):
    if file.endswith(".jpg"):
        im = imread("./train/" + file)
        im = im.flatten() # flatten im into a vector
        images.append(im) 
        filename.append(file)
A_pp = np.stack(images).T # build a matrix where each column is a flattened image
print(A_pp.shape)
plt.imshow(A_pp[:, 0].reshape(dimension))
plt.show()




#preprocessing of image
def preprocess(A_pp):
    # YOUR CODE HERE
    
    #shape of input A_pp
    d,n = np.shape(A_pp) # d = 24576, n = 199
  
    #A_means
    A_means = np.zeros(d)
    for i in range (d):
        A_means[i] = np.average(A_pp[i])
        
        
    #Q_norms
    Q = np.zeros((d,n))
    for i in range (d):
        Q[i,:] = A_pp[i,:] / A_means[i]
        
    Q_norms = np.zeros(d)
    for i in range (d):
        Q_norms[i] = max(abs(Q[i,:]))
        
    #A
    A = np.zeros((d,n))
    for i in range (d):
        A[i,:] = Q[i,:] / Q_norms[i]
    
    return A, Q_norms, A_means
    pass
    
    

A, Q_norms, A_means = preprocess(A_pp)
print(A)
print(Q_norms)
print(A_means)




#get eigenvalues
def eigen_ped(A):
    # YOUR CODE HERE
    
    #shape of input A
    d,n = np.shape(A) # d = 24576, n = 199
    
    #covarianve matrix A@At
    S = A.T @ A
    
    #eigenvalues and eigenvectors of S
    val, vec = np.linalg.eig(S) 
    
    #the non-zero eigenvalues of AtA and AAt are the same
    D = np.zeros(d)
    for i in range (n):
        D[i] = val[i]
    
    #eigenvector of AAt is the eigenvector of AtA left multiplied by A
    F = A @ vec
    
    #normalise F for orthogonality @518
    norm =  np.linalg.norm(F, axis = 0)
    for i in range(n):
        F[:,i] = F[:,i] / norm[i]
    
    return F, D
    pass
    
    

#For the purposes of doing this assignment, this code isn't really here. Pretend it's engraved in rock.
F, D = eigen_ped(A)
F_real = np.real(F)
print('Orthogonality Check (should be close to 0): ', F_real[:, 0].T@F_real[:, 1])
print('Unit Vector Check: ', math.isclose(np.linalg.norm(F_real[:,0]), 1))
print(F.shape) # It should be (24576, 199)
print(D.shape) # It should be (24576)

# The visulisation of an Eigen Pedestrain should **look like** a pedestrain.
print('Visualise an Eigen Pedestrain:')
ep = np.rint((F[:,0] * Q_norms + A_means).reshape(dimension)).astype(int)
plt.imshow(ep)
plt.show()




#dimensionality reduction
def reduce_dimensionality(image_vector, k, F, D, A_means, Q_norms):
    # YOUR CODE HERE
    
    #length of image_vector is 24576
    #shape of input F
    d,n = np.shape(F) # d = 24576, n = 199
    
    #p: how much you capture
    p = np.sum(D[:k]) / np.sum(D)
    
    #pre-process image_vector
    img = (image_vector - A_means) / Q_norms
    
    #compressed image
    compressed_image = np.zeros(n)
    z = F[:,:k].T @ img
    for i in range (k):
        compressed_image[i] = z[i]

    
    return compressed_image, p

# Display Code. Leave it alooooooooooone.
Idx = 0
compressed_image, p = reduce_dimensionality(A_pp[:, Idx], 80, F, D, A_means, Q_norms)
print(compressed_image.shape) # should be (199,)
print('Variance Captured:', int(p * 100), '%')




#reconstruction of reduced image
def reconstruct_image(compressed_image, F, Q_norms, A_means):
    # YOUR CODE HERE
    
    #Rp: reverse dimensionaility reduction
    Rp = compressed_image @ F.T
    
    #R: full reverse
    R = (Rp * Q_norms + A_means).astype(int) #reverse pre-processing and typecast to integer since RGB = [0,255]
    R = R.reshape((128,64,3)) #reshape
    
    return R
    
    
    
    
#Display Code. Leave it alooooooooooone.
R_c = reconstruct_image(compressed_image, F, Q_norms, A_means)
print('Compressed Image: ')
plt.imshow(R_c)
plt.show()
Img = A[:, Idx]
R_o = A_pp[:, Idx].reshape(dimension)
print('Original Image')
plt.imshow(R_o)
plt.show()




#find the most similar image
def the_nearest_image(query_image, gallery_images, k, F, D, A_means, Q_norms):
    # YOUR CODE HERE
    
    #shape(query_image) = (24576,)
    d, n = np.shape(gallery_images) # d = 24576, n = 90
    
    #compressed query image
    cQI, p = reduce_dimensionality(query_image, k, F, D, A_means, Q_norms)
    
    #euclidean distance between vectors of compressed images
    distance = np.zeros(n)
    for i in range (n):
        cGI, p = reduce_dimensionality(gallery_images[:,i], k, F, D, A_means, Q_norms) #compressed gallery image
        distance[i] = np.linalg.norm(cQI - cGI) #euclidean distance

    #find argmin
    ini = np.argmin(distance)
    
    return ini

# Display Code. Leave it alooooooooooone.
# read a query image
query_image = imread("./val_query/0227_c2s1_046476_01.jpg")
query_image = query_image.flatten()

# read gallery images
gallery_images = []
original_gallery_images = []
filename = []
for file in os.listdir("./gallery"):
    if file.endswith(".jpg"):
        im = imread("./gallery/" + file)
        original_gallery_images.append(im)
        im = im.flatten() # flatten im into a vector
        gallery_images.append(im) 
        filename.append(file)
        
original_gallery_images = np.array(original_gallery_images)
gallery_images = np.stack(gallery_images).T

idx = the_nearest_image(query_image, gallery_images, 80, F, D, A_means, Q_norms)
plt.imshow(query_image.reshape(dimension))
plt.show()
plt.imshow(gallery_images[:, idx].reshape(dimension))
plt.show()




#get similarity ranking
def image_similarity_ranking(image_gallery, image_query):
    # YOUR CODE HERE
    
    #set k
    k = 80
    
    #shape of inputs
    # n = number of images in the gallery = 90
    # r = rows in an image = 128
    # c = columns in an image = 64
    #rgb = colour vector = 3
    n, r, c, rgb = np.shape(image_gallery)
    d = r * c * rgb # dimension = 24576    
    
    #reshape
    imgal = np.zeros((d,n))
    for i in range (n):
        imgal[:,i] = image_gallery[i,:].flatten()
    image_query = image_query.flatten()    
    
    #pre-process
    A, Q_norms, A_means = preprocess(imgal)
    
    #find eigenvalues and eigenvectors
    F, D = eigen_ped(A)
    
    #compressed query image
    cQI, p = reduce_dimensionality(image_query, k, F, D, A_means, Q_norms)
    
    #euclidean distance between vectors of compressed images
    distance = np.zeros(n)
    for i in range (n):
        cGI, p = reduce_dimensionality(imgal[:,i], k, F, D, A_means, Q_norms) #compressed gallery image
        distance[i] = np.linalg.norm(cQI - cGI) #euclidean distance
        
    #sort indexes
    idx = np.argsort(distance)

    return idx

# Display Code. Leave it alooooooooooone.

id_list = image_similarity_ranking(original_gallery_images, imread("./val_query/0227_c2s1_046476_01.jpg"))

plt.imshow(imread("./val_query/0227_c2s1_046476_01.jpg"))
plt.show()
plt.imshow(original_gallery_images[id_list[0]])
plt.show()

