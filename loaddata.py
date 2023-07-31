import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split



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
    # print(v_data.shape)
    # print(v_label.shape)
    """
    # print("yes")
    v_data = v_data[:1954,:]
    n_data = n_data[:1954,:]
    r_data = r_data[:1954,:]
    l_data = l_data[:1954,:]
    a_data = a_data[:1954,:]
    d_data = d_data[:1954,:]

    for i in range(1950):
        if v_label[i] >= v_data[i]:
            break
        v_label_2.append(v_label[i])
    for i in range(1950):
        if n_label[i] >= n_data[i]:
            break
        n_label_2.append(n_label[i])
    for i in range(1950):
        if l_label[i] >= l_data[i]:
            break
        l_label_2.append(l_label[i])
    for i in range(1950):
        if r_label[i] >= r_data[i]:
            break
        r_label_2.append(r_label[i])
    for i in range(1950):
        if a_label[i] >= a_data[i]:
            break
        a_label_2.append(a_label[i])
    for i in range(1950):
        if d_label[i] >= d_data[i]:
            break
        d_label_2.append(d_label[i])
    v_label = v_label_2
    n_label = n_label_2
    r_label = r_label_2
    l_label = l_label_2
    a_label = a_label_2
    d_label = d_label_2
    # print(v_data.shape,n_data.shape,r_data.shape,l_data.shape,a_data.shape,d_data.shape)
    """
    v_train_data,v_test_data,v_train_label,v_test_label = train_test_split(v_data,v_label,random_state = 42,train_size = 0.6)
    n_train_data,n_test_data,n_train_label,n_test_label = train_test_split(n_data,n_label,random_state = 42,train_size = 0.6)
    l_train_data,l_test_data,l_train_label,l_test_label = train_test_split(l_data,l_label,random_state = 42,train_size = 0.6)
    r_train_data,r_test_data,r_train_label,r_test_label = train_test_split(r_data,r_label,random_state = 42,train_size = 0.6)
    a_train_data,a_test_data,a_train_label,a_test_label = train_test_split(a_data,a_label,random_state = 42,train_size = 0.6)
    d_train_data,d_test_data,d_train_label,d_test_label = train_test_split(d_data,d_label,random_state = 42,train_size = 0.6)

    # print(v_train_data.shape,v_test_data.shape,v_train_label.shape,v_test_label.shape,"v's data and label")
    # print(n_train_data.shape,n_test_data.shape,n_train_label.shape,n_test_label.shape,"n's data and label")
    # print(l_train_data.shape,l_test_data.shape,l_train_label.shape,l_test_label.shape,"l's data and label")
    # print(r_train_data.shape,r_test_data.shape,r_train_label.shape,r_test_label.shape,"r's data and label")
    v_train_label = v_train_label.reshape(v_train_label.shape[0],1)
    n_train_label = n_train_label.reshape(n_train_label.shape[0],1)
    l_train_label = l_train_label.reshape(l_train_label.shape[0],1)
    r_train_label = r_train_label.reshape(r_train_label.shape[0],1)
    a_train_label = a_train_label.reshape(a_train_label.shape[0],1)
    d_train_label = d_train_label.reshape(d_train_label.shape[0],1)
    v_test_label = v_test_label.reshape(v_test_label.shape[0],1)
    n_test_label = n_test_label.reshape(n_test_label.shape[0],1)
    l_test_label = l_test_label.reshape(l_test_label.shape[0],1)
    r_test_label = r_test_label.reshape(r_test_label.shape[0],1)
    a_test_label = a_test_label.reshape(a_test_label.shape[0],1)
    d_test_label = d_test_label.reshape(d_test_label.shape[0],1)

    sum_train_data = np.vstack((v_train_data,n_train_data))
    sum_train_data = np.vstack((sum_train_data,l_train_data))
    sum_train_data = np.vstack((sum_train_data,r_train_data))
    sum_train_data = np.vstack((sum_train_data,a_train_data))
    sum_train_data = np.vstack((sum_train_data,d_train_data))
    # print(sum_train_data.shape)
    sum_train_label = np.vstack((v_train_label,n_train_label))
    sum_train_label = np.vstack((sum_train_label,l_train_label))
    sum_train_label = np.vstack((sum_train_label,r_train_label))
    sum_train_label = np.vstack((sum_train_label,a_train_label))
    sum_train_label = np.vstack((sum_train_label,d_train_label))
    # print(sum_train_label.shape)
    sum_test_data = np.vstack((v_test_data,n_test_data))
    sum_test_data = np.vstack((sum_test_data,l_test_data))
    sum_test_data = np.vstack((sum_test_data,r_test_data))
    sum_test_data = np.vstack((sum_test_data,a_test_data))
    sum_test_data = np.vstack((sum_test_data,d_test_data))
    # print(sum_test_data.shape)
    # print(v_test_label.shape,n_test_label.shape)
    sum_test_label = np.vstack((v_test_label,n_test_label))
    sum_test_label = np.vstack((sum_test_label,l_test_label))
    sum_test_label = np.vstack((sum_test_label,r_test_label))
    sum_test_label = np.vstack((sum_test_label,a_test_label))
    sum_test_label = np.vstack((sum_test_label,d_test_label))
    # print(sum_test_label.shape)
    return sum_test_data,sum_test_label,sum_train_data,sum_train_label

def filtered_data():
    v_data  = np.loadtxt("v_data.txt",delimiter=",")
    v_label = np.loadtxt("v_label.txt",dtype = str)
    n_data  = np.loadtxt("n_data.txt",delimiter=",")
    n_label = np.loadtxt("n_label.txt",dtype = str)
    l_data  = np.loadtxt("r_data.txt",delimiter=",")
    l_label = np.loadtxt("r_label.txt",dtype = str)
    r_data  = np.loadtxt("l_data.txt",delimiter=",")
    r_label = np.loadtxt("l_label.txt",dtype = str)
    a_data  = np.loadtxt("A_data.txt",delimiter=",")
    a_label = np.loadtxt("A_label.txt",dtype = str)
    d_data  = np.loadtxt("D_data.txt",delimiter=",")
    d_label = np.loadtxt("D_label.txt",dtype = str)

    return v_data,v_label,n_data,n_label,l_data,l_label,r_data,r_label,a_data,a_label,d_data,d_label


def feature_data():
    v_label_point = np.loadtxt(r"C:\Users\ASUS\OneDrive\桌面\python\course\mit-bih-arrhythmia-database-1.0.0\V_label_point")
    n_label_point = np.loadtxt(r"C:\Users\ASUS\OneDrive\桌面\python\course\mit-bih-arrhythmia-database-1.0.0\N_label_point")
    r_label_point = np.loadtxt(r"C:\Users\ASUS\OneDrive\桌面\python\course\mit-bih-arrhythmia-database-1.0.0\R_label_point")
    l_label_point = np.loadtxt(r"C:\Users\ASUS\OneDrive\桌面\python\course\mit-bih-arrhythmia-database-1.0.0\L_label_point")
    a_label_point = np.loadtxt(r"C:\Users\ASUS\OneDrive\桌面\python\course\mit-bih-arrhythmia-database-1.0.0\A_label_point")
    d_label_point = np.loadtxt(r"C:\Users\ASUS\OneDrive\桌面\python\course\mit-bih-arrhythmia-database-1.0.0\D_label_point")
    print("yes")

    v_label_point = v_label_point.reshape(v_label_point.shape[0],1)
    n_label_point = n_label_point.reshape(n_label_point.shape[0],1)
    l_label_point = l_label_point.reshape(l_label_point.shape[0],1)
    r_label_point = r_label_point.reshape(r_label_point.shape[0],1)
    a_label_point = a_label_point.reshape(a_label_point.shape[0],1)
    d_label_point = d_label_point.reshape(d_label_point.shape[0],1)
    print(v_label_point.shape)
    print(n_label_point.shape)
    sum_label_point = np.vstack((v_label_point,n_label_point))
    print(sum_label_point.shape)
    sum_label_point = np.vstack((sum_label_point,l_label_point))
    sum_label_point = np.vstack((sum_label_point,r_label_point))
    sum_label_point = np.vstack((sum_label_point,a_label_point))
    sum_label_point = np.vstack((sum_label_point,d_label_point))
    
    return sum_label_point