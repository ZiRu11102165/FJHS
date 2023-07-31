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
    v_data  = np.loadtxt("v_data.txt",delimiter=",")
    v_label = np.loadtxt("v_label.txt",dtype = str)
    n_data  = np.loadtxt("n_data.txt",delimiter=",")
    n_label = np.loadtxt("n_label.txt",dtype = str)
    r_data  = np.loadtxt("r_data.txt",delimiter=",")
    r_label = np.loadtxt("r_label.txt",dtype = str)
    l_data  = np.loadtxt("l_data.txt",delimiter=",")
    l_label = np.loadtxt("l_label.txt",dtype = str)
    a_data  = np.loadtxt("A_data.txt",delimiter=",")
    a_label = np.loadtxt("A_label.txt",dtype = str)
    d_data  = np.loadtxt("D_data.txt",delimiter=",")
    d_label = np.loadtxt("D_label.txt",dtype = str)


    v_label = v_label.reshape(v_label.shape[0],1)
    n_label = n_label.reshape(n_label.shape[0],1)
    r_label = r_label.reshape(r_label.shape[0],1)
    l_label = l_label.reshape(l_label.shape[0],1)
    a_label = a_label.reshape(a_label.shape[0],1)
    d_label = d_label.reshape(d_label.shape[0],1)
    # print("n_data",n_data.shape)
    n_data_1, n_test_data,n_label_1, n_test_label = train_test_split(n_data, n_label, test_size=0.1, shuffle=False)    
    v_data_1, v_test_data,v_label_1, v_test_label = train_test_split(v_data, v_label, test_size=0.1, shuffle=False)
    r_data_1, r_test_data,r_label_1, r_test_label = train_test_split(r_data, r_label, test_size=0.1, shuffle=False)
    l_data_1, l_test_data,l_label_1, l_test_label = train_test_split(l_data, l_label, test_size=0.1, shuffle=False)
    a_data_1, a_test_data,a_label_1, a_test_label = train_test_split(a_data, a_label, test_size=0.1, shuffle=False)
    d_data_1, d_test_data,d_label_1, d_test_label = train_test_split(d_data, d_label, test_size=0.1, shuffle=False)
    print(n_data_1.shape,n_test_data.shape)

    n_data = n_data_1
    v_data = v_data_1
    r_data = r_data_1
    l_data = l_data_1
    a_data = a_data_1
    d_data = d_data_1
    n_label = n_label_1
    v_label = v_label_1
    r_label = r_label_1
    l_label = l_label_1
    a_label = a_label_1
    d_label = d_label_1

    # print("n_data_non_shuffle",n_data.shape)
    # print(v_data.shape)
    # print(r_data.shape)
    # print(l_data.shape)
    # print(a_data.shape)
    # print(d_data.shape)
    abnormal_data = np.vstack((v_data,r_data))
    abnormal_data = np.vstack((abnormal_data,l_data))
    abnormal_data = np.vstack((abnormal_data,a_data))
    abnormal_data = np.vstack((abnormal_data,d_data))

    normal_data = n_data

    abnormal_label = np.vstack((v_label,r_label))
    abnormal_label = np.vstack((abnormal_label,l_label))
    abnormal_label = np.vstack((abnormal_label,a_label))
    abnormal_label = np.vstack((abnormal_label,d_label))

    normal_label = n_label

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

    # print(sum_train_data.shape,sum_test_data.shape,sum_train_label.shape,sum_test_label.shape)

    sum_train_label = []
    sum_test_label = []
    # print("len_n",len(normal_train_data))
    for i in range(len(normal_train_data)):
        sum_train_label.append("N")
    for i in range(len(abnormal_train_data)):
        sum_train_label.append("b")    
    for i in range(len(normal_test_data)):
        sum_test_label.append("N")
    for i in range(len(abnormal_test_data)):
        sum_test_label.append("b")    
    sum_train_label = np.array(sum_train_label)
    sum_test_label = np.array(sum_test_label)
    print(sum_train_data.shape,sum_test_data.shape,sum_train_label.shape,sum_test_label.shape)

    return sum_test_data,sum_test_label,sum_train_data,sum_train_label
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(sum_train_data, sum_train_label)
predicted = clf.predict(sum_test_data)
accuracy = clf.score(sum_test_data, sum_test_label)
print(accuracy)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(sum_test_label,predicted)
plot_confusion_matrix(clf, sum_test_data, sum_test_label)  
plt.show() 
a1 = sns.heatmap(cm,square=True,annot=True,fmt='d',linecolor='white',cmap='RdBu',linewidths=1.5,cbar=False,ax=axes[0, 2])
a1.set_ylabel('true')
a1.set_xlabel('predict')
a1.set_title('RF')
print("3")
"""

v_data  = np.loadtxt("v_data.txt",delimiter=",")
v_label = np.loadtxt("v_label.txt",dtype = str)
n_data  = np.loadtxt("n_data.txt",delimiter=",")
n_label = np.loadtxt("n_label.txt",dtype = str)
r_data  = np.loadtxt("r_data.txt",delimiter=",")
r_label = np.loadtxt("r_label.txt",dtype = str)
l_data  = np.loadtxt("l_data.txt",delimiter=",")
l_label = np.loadtxt("l_label.txt",dtype = str)
a_data  = np.loadtxt("A_data.txt",delimiter=",")
a_label = np.loadtxt("A_label.txt",dtype = str)
d_data  = np.loadtxt("D_data.txt",delimiter=",")
d_label = np.loadtxt("D_label.txt",dtype = str)


v_label = v_label.reshape(v_label.shape[0],1)
n_label = n_label.reshape(n_label.shape[0],1)
r_label = r_label.reshape(r_label.shape[0],1)
l_label = l_label.reshape(l_label.shape[0],1)
a_label = a_label.reshape(a_label.shape[0],1)
d_label = d_label.reshape(d_label.shape[0],1)
# print("n_data",n_data.shape)
n_data_1, n_test_data,n_label_1, n_test_label = train_test_split(n_data, n_label, test_size=0.1, shuffle=False)    
v_data_1, v_test_data,v_label_1, v_test_label = train_test_split(v_data, v_label, test_size=0.1, shuffle=False)
r_data_1, r_test_data,r_label_1, r_test_label = train_test_split(r_data, r_label, test_size=0.1, shuffle=False)
l_data_1, l_test_data,l_label_1, l_test_label = train_test_split(l_data, l_label, test_size=0.1, shuffle=False)
a_data_1, a_test_data,a_label_1, a_test_label = train_test_split(a_data, a_label, test_size=0.1, shuffle=False)
d_data_1, d_test_data,d_label_1, d_test_label = train_test_split(d_data, d_label, test_size=0.1, shuffle=False)
print(n_data_1.shape,n_test_data.shape)
    