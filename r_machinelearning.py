import numpy as np
import matplotlib.pyplot as plt
import read_data
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
from sklearn import metrics
from sklearn.metrics import accuracy_score
import loaddata
from sklearn.metrics import accuracy_score, average_precision_score,precision_score,f1_score,recall_score
import seaborn as sns
import matplotlib.pyplot as plt  
from sklearn.datasets import make_classification
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

def data():
    r_data  = np.loadtxt("r_data.txt",delimiter=",")
    r_label = np.loadtxt("r_label.txt",dtype = str)
    a_data  = np.loadtxt("A_data.txt",delimiter=",")
    a_label = np.loadtxt("A_label.txt",dtype = str)
    d_data  = np.loadtxt("D_data.txt",delimiter=",")
    d_label = np.loadtxt("D_label.txt",dtype = str)



    r_label = r_label.reshape(r_label.shape[0],1)
    a_label = a_label.reshape(a_label.shape[0],1)
    d_label = d_label.reshape(d_label.shape[0],1)
    
    r_data_1, r_test_data,r_label_1, r_test_label = train_test_split(r_data, r_label, test_size=0.1, shuffle=False)
    a_data_1, a_test_data,a_label_1, a_test_label = train_test_split(a_data, a_label, test_size=0.1, shuffle=False)
    d_data_1, d_test_data,d_label_1, d_test_label = train_test_split(d_data, d_label, test_size=0.1, shuffle=False)

    r_data = r_data_1
    a_data = a_data_1
    d_data = d_data_1
    r_label = r_label_1
    a_label = a_label_1
    d_label = d_label_1

    abnormal_data = np.vstack((d_data,a_data))

    normal_data = r_data

    abnormal_label = np.vstack((d_label,a_label))

    normal_label = r_label

    # abnormal_data = abnormal_data.reshape(abnormal_data.shape[0],1)
    # normal_data = normal_data.reshape(normal_data.shape[0],1)
    # abnormal_label = abnormal_label.reshape(abnormal_label.shape[0],1)
    # normal_label = normal_label.reshape(normal_label.shape[0],1)

    normal_train_data,normal_test_data,normal_train_label,normal_test_label = train_test_split(normal_data,normal_label,random_state = 42,train_size = 0.8)
    abnormal_train_data,abnormal_test_data,abnormal_train_label,abnormal_test_label = train_test_split(abnormal_data,abnormal_label,random_state = 42,train_size = 0.8)

    sum_train_data = np.vstack((normal_train_data,abnormal_train_data))
    sum_test_data = np.vstack((normal_test_data,abnormal_test_data))
    sum_train_label = np.vstack((normal_train_label,abnormal_train_label))
    sum_test_label = np.vstack((normal_test_label,abnormal_test_label))

    print(sum_train_data.shape,sum_test_data.shape,sum_train_label.shape,sum_test_label.shape)

    sum_train_label = []
    sum_test_label = []

    for i in range(len(normal_train_data)):
        sum_train_label.append("R")
    for i in range(len(abnormal_train_data)):
        sum_train_label.append("b")    
    for i in range(len(normal_test_data)):
        sum_test_label.append("R")
    for i in range(len(abnormal_test_data)):
        sum_test_label.append("b")    
    sum_train_label = np.array(sum_train_label)
    sum_test_label = np.array(sum_test_label)
    print(sum_train_data.shape,sum_test_data.shape,sum_train_label.shape,sum_test_label.shape)

    return sum_test_data,sum_test_label,sum_train_data,sum_train_label
