import cv2
from cv2 import *
import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import skimage
from skimage.transform import resize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm
import glob
import cv2 as cv
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import metrics



train_dir = "E:\\ASL_Alphabet_Dataset\\asl_alphabet_train"
test_dir =  "E:\\ASL_Alphabet_Dataset\\asl_alphabet_test"
thisdict = {"A_test.jpg": 0,"B_test.jpg": 1,"C_test.jpg": 2,"D_test.jpg": 3,"E_test.jpg":4,"F_test.jpg":5,"G_test.jpg":6,"H_test.jpg":7,"I_test.jpg":8,"J_test.jpg":9,"K_test.jpg":10,"L_test.jpg":11,"M_test.jpg":12,"N_test.jpg":13,"nothing_test.jpg":14,"O_test.jpg":15,"P_test.jpg":16,"Q_test.jpg":17,"R_test.jpg":18,"S_test.jpg":19,"space_test.jpg":20,"T_test.jpg":21,"U_test.jpg":22,"V_test.jpg":23,"W_test.jpg":24,"X_test.jpg":25,"Y_test.jpg":26,"Z_test.jpg":27,"del_test.jpg":28,"A": 0,"B": 1,"C": 2,"D": 3,"E":4,"F":5,"G":6,"H":7,"I":8,"J":9,"K":10,"L":11,"M":12,"N":13,"nothing":14,"O":15,"P":16,"Q":17,"R":18,"S":19,"space":20,"T":21,"U":22,"V":23,"W":24,"X":25,"Y":26,"Z":27,"del":28}


def load_train(folder):
    images = []
    Y=[]
    ListOfimages=[]
    for foldername in os.listdir(folder):
        ListOfimages=os.listdir("E:\\ASL_Alphabet_Dataset\\asl_alphabet_train\\"+foldername)
        for image_filename in range(len(ListOfimages)):
                if image_filename>5500 and image_filename<6500:
                    img_file = cv2.imread(os.path.join(folder,foldername,ListOfimages[image_filename]))
                    if img_file is not None:
                        img_file=cv2.resize(img_file,(200,200))
                        img_file=cv2.resize(img_file, (0,0), fx=0.25, fy=0.25)
                        images.append(img_file)
                        Y.append(thisdict[foldername])
    return images,Y


X_, Y_ = load_train(train_dir) 

#len(X_)
'''
cv2.imshow('Original img',X_[50])
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

def load_test(folder):
    images = []
    Y=[]
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            img=cv2.resize(img,(200,200))
            img = cv2.resize(img, (0,0), fx=0.25, fy=0.25)
            images.append(img)
            Y.append(thisdict[filename])
            
    return images,Y


Xtest,Ytest= load_test(test_dir)

#len(Xtest)

def processing(X,Y):    
   X_RGB=[]
   X_Grey=[]
   X_Binary=[]
   X_RGB_BLUR=[]
   X_Grey_BLUR=[]
   X_Binary_BLUR=[]
   edges1=[]
   edges2=[]
   edges3=[]
   F_edges1=[]
   F_edges2=[]
   F_edges3=[]
   y=[]
   for image in X:
        X_RGB.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        X_Grey.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        
   for image in  X_Grey:
        r, binary=cv2.threshold(image, 125, 255, cv2.THRESH_BINARY_INV)
        X_Binary.append(binary) 
        
   for image in range(len(X)):
        X_RGB_BLUR.append(cv2.GaussianBlur(X_RGB[image], (7, 7), 0))
        X_Grey_BLUR.append(cv2.GaussianBlur(X_Grey[image], (7, 7), 0)) 
        X_Binary_BLUR.append(cv2.GaussianBlur(X_Binary[image], (7, 7), 0))

   for img in range(len(X)):
        edges1.append( cv2.Canny(image=X_RGB_BLUR[img], threshold1=4, threshold2=100))
        edges2.append( cv2.Canny(image=X_Grey_BLUR[img], threshold1=4, threshold2=100))
        edges3.append( cv2.Canny(image=X_Binary_BLUR[img], threshold1=4, threshold2=100))
   edges1=np.asarray(edges1)
   edges2=np.asarray(edges2)
   edges3=np.asarray(edges3) 
   Y= np.asarray(Y)
   for i in range(len(X)):
       F_edges1.append(edges1[i].flatten())
       F_edges2.append(edges2[i].flatten())
       F_edges3.append(edges3[i].flatten())
       y.append(Y[i].flatten())
   return F_edges1,F_edges2,F_edges3,y
    

#RGB_train,Grey_train,Binary_train,Y_train= processing(X_, Y_)
x_RGB_train,x_Grey_train,x_Binary_train,Y_train= processing(X_, Y_)
x_RGB_test,x_Grey_test,x_Binary_test,Y_test= processing(Xtest, Ytest)


#################################### to know dimintion
x_RGB_train=np.array(x_RGB_train)
x_Grey_train=np.array(x_Grey_train)
x_Binary_train=np.array(x_Binary_train)
Y_train=np.array(Y_train)

x_RGB_test=np.array(x_RGB_test)
x_Grey_test=np.array(x_Grey_test)
x_Binary_test=np.array(x_Binary_test)
Y_test=np.array(Y_test)

#################################### to test dimintions
x_RGB_test.shape
x_RGB_train.shape

len(x_RGB_train.shape)
len(x_RGB_test.shape)

len(Y_test.shape)

x_RGB_test.ndim
#################################### to show image
cv2.imshow('Original img',x_Grey_train[5])
cv2.waitKey(0)
cv2.destroyAllWindows()

###############################################################################
############################# Classifiers #####################################
###############################################################################
def decisionTree(X_train,Y_train,X_test,Y_test) :
  #Create Decision Tree classifer object
  clf = DecisionTreeClassifier()
 # Train Decision Tree Classifer
  clf = clf.fit(X_train,Y_train)
 #Predict the response for test dataset
  y_pred = clf.predict(X_test)
  # Model Accuracy, how often is the classifier correct?
  print("Accuracy:  %.2f%%" % (metrics.accuracy_score(Y_test, y_pred)*100))
  print ('Recall:', recall_score(Y_test,  y_pred, average='micro'))
  print ('Precision:', precision_score(Y_test, y_pred, average='micro'))
   
############
def SVM(X_train,Y_train,X_test,Y_test):
    #Create a svm Classifier
   clf = svm.SVC(kernel='linear') # Linear Kernel
#Train the model using the training sets
   clf.fit(X_train, Y_train)
#Predict the response for test dataset
   y_pred = clf.predict(X_test)
#Accuracy
   print("Accuracy:  %.2f%%" % (metrics.accuracy_score(Y_test, y_pred)*100))
   print ('Recall:', recall_score(Y_test,  y_pred, average='micro'))
   print ('Precision:', precision_score(Y_test, y_pred, average='micro'))
 
############
def RForest(X_train,Y_train,X_test,Y_test):
    r_forest = RandomForestClassifier(max_depth=100, random_state=0) #classifer
    r_forest = r_forest.fit(X_train,Y_train) #fit classifer
    y_pred = r_forest.predict(X_test) #Y predict
    print('Accuracy : %.3f' % (metrics.accuracy_score(Y_test, y_pred)*100))
    print('Precision: %.3f' % precision_score(Y_test, y_pred, average='micro'))
    print('Recall: %.3f' % recall_score(Y_test, y_pred, average='micro'))

###############################################################################
######################### Inputs ##############################################
###############################################################################
#input RGB
RForest(x_RGB_train,Y_train,x_RGB_test,Y_test)
SVM(x_RGB_train,Y_train,x_RGB_test,Y_test)
decisionTree(x_RGB_train,Y_train,x_RGB_test,Y_test)
  
#input Grey
RForest(x_Grey_train,Y_train,x_Grey_test,Y_test)
SVM(x_Grey_train,Y_train,x_Grey_test,Y_test)
decisionTree(x_Grey_train,Y_train,x_Grey_test,Y_test)
    

#input Binary
RForest(x_Binary_train,Y_train,x_Binary_test,Y_test)
SVM(x_Binary_train,Y_train,x_Binary_test,Y_test)
decisionTree(x_Binary_train,Y_train,x_Binary_test,Y_test)
