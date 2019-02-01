# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 11:52:39 2018

@author: rendla
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 11:20:31 2018

@author: rendla
"""
#importing the required libraries
import cv2
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load data into the corresponding variables
dataset = fetch_lfw_people(min_faces_per_person=100)
_, h, w = dataset.images.shape
X = dataset.data
y = dataset.target
names = dataset.target_names

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) #splitting of train and test data
n_components = 100  #assigning the reduced dimension

#this funtion normalizes the given data
def fnormal(X):
    mu=np.array(sc.mean(X,0)) #get the mean of the data
    var=np.array(sc.var(X,0)) #get the variance of the data
    var=var**0.5 #calculate the standard deviation of the datat
    mu=np.ones([np.size(X,0),np.size(X,1)])*mu #build the mean matrix
    var=np.ones([np.size(X,0),np.size(X,1)])*var #build the standard deviation matrix
    X=np.subtract(X,mu) #subtract the mean from the given data
    X=np.divide(X,var)  #divide the data with its corresponding standard deviation
    return X,mu,var #return the normalised data along with mean and standard deviation

#this function calculates and returns the eigen vectors of given data 
def pca(X):
    X_norm,mu,sd=fnormal(X)  #get the normalised data along with mean and standard deviation  
    X_norm=np.matrix(X_norm) 
    cov=np.transpose(X_norm)*X_norm 
    cov=cov/np.size(X,0)    #get the covarince matrix
    (U,V)=np.linalg.eigh(cov) #calculate the eigen values and eigen vectors
    V=V.T                 #this takes the transpose of the eigen vecotrs, so that each row corresponds to a eigen vector
    return U,V

#this function projects the given high dimensional data matrix onto a lower dimension 'K'
def projecter(X,V,k):
    V=np.matrix(V)                  
    X=np.matrix(X)
    X_pca=np.matrix(X*V[-k:].T)    #projecting the data on lower dimension 'k'
    return X_pca                  #return the projected data
U,V=pca(X_train)              #collecting the eigen values, eigen vectors into variables U and V repectively

#function to calculate the average accuracy by changing the value of n_components  
t=0           #this variable used to store the average accuracy
def acccal(n_components,X_train,y_train,V,X_test,y_test):
    X_train_pca=projecter(X_train,V,n_components) #this stores the resulting train data after dimensionality reduction
    X_test_pca=projecter(X_test,V,n_components)   #this stores the resulting test data after dimensionality reduction
    print(n_components-100)                      #in the below line we train the nueral network using X_train and labels y_train 
    trainer = MLPClassifier(hidden_layer_sizes=(1024,), batch_size=256, early_stopping=True).fit(X_train_pca, y_train)
    y_pred = trainer.predict(X_test_pca)         #predict using the trained nueral network by passing test_pca to it      
    t=accuracy_score(y_test,y_pred)              #calculating the accuracy of the prediction 
    return t                              #return the accuracy_score

X_train_pca=projecter(X_train,V,130) #this stores the resulting train data after dimensionality reduction
X_test_pca=projecter(X_test,V,n_components)   #this stores the resulting test data after dimensionality reduction
trainer = MLPClassifier(hidden_layer_sizes=(1024,), batch_size=256, early_stopping=True).fit(X_train_pca, y_train)# to make sure trainer variable is available gloabally, it is written outside the funciton acccal
b=0 #stores the maximum accuracy
c=0 #stores the n_compnents at maximum accuracy
for i in range(50):
    u=acccal(n_components+i,X_train,y_train,V,X_test,y_test)
    if(b<u):
        b=u
        c=i+100
    t=t+u #accumulate the accuracy in to variable t
print("The resulting accuracy is %f" %(t/50.0)) #printing the resulting accuracy
print("The maximum accuracy is %f at n_components=%d" %(b,c)) #printing the resulting accuracy    
#this loop is to print the final top 5 eigen faces
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(V[-i-1].reshape(62,47),cmap="gray")
#Predicting the label of an image
def new_predictor(f_path,V):
    image = cv2.imread(f_path,0)  #read the image into image variable
    image1 = cv2.resize(image,(62,47)) #resize the image 
    image2 = image1.flatten()
    projected_img = projecter(image2,V,130)
    predicted = trainer.predict(projected_img)
    return str(names[predicted[0]])

#image_path = "CroppedYale/yaleB15/yaleB15_P00A+000E+00.pgm"
img_path = input()
label = new_predictor(img_path,V)
print("\nGiven input face belongs to "+label)