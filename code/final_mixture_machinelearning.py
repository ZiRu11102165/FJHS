import numpy as np
import matplotlib.pyplot as plt
import read_data
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
from sklearn import metrics
from sklearn.metrics import accuracy_score
import loaddata
from sklearn.metrics import accuracy_score, average_precision_score,precision_score,f1_score,recall_score
import seaborn as sns
import n_machinelearning
import v_machinelearning
import l_machinelearning
import r_machinelearning
import d_machinelearning
import a_machinelearning
import a_r_feature_machinelearning
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report  # 全部指標的report
import pandas as pd
import neurokit2 as nk

def find_elements(array):
    elements = set()

    for item in array:
        elements.add(tuple(item))  # Convert NumPy arrays to tuples

    return elements


data = n_machinelearning.data()
sum_test_data = data[0]
sum_test_label = data[1]
sum_train_data = data[2]
sum_train_label = data[3]

clf_n = ExtraTreesClassifier(n_estimators=100, random_state=0)
clf_n.fit(sum_train_data, sum_train_label)
print("finish fitting n")

data = v_machinelearning.data()
sum_test_data = data[0]
sum_test_label = data[1]
sum_train_data = data[2]
sum_train_label = data[3]

knn_v = KNeighborsClassifier(n_neighbors=1)
knn_v.fit(sum_train_data, sum_train_label)
print("finish fitting v")


data = l_machinelearning.data()
sum_test_data = data[0]
sum_test_label = data[1]
sum_train_data = data[2]
sum_train_label = data[3]


knn_l = KNeighborsClassifier(n_neighbors=1)
knn_l.fit(sum_train_data, sum_train_label)
print("finish fitting l")


data = r_machinelearning.data()
sum_test_data = data[0]
sum_test_label = data[1]
sum_train_data = data[2]
sum_train_label = data[3]

clf_r = ExtraTreesClassifier(n_estimators=100, random_state=0)
clf_r.fit(sum_train_data, sum_train_label)
print("finish fitting r")

data = d_machinelearning.data()
sum_test_data = data[0]
sum_test_label = data[1]
sum_train_data = data[2]
sum_train_label = data[3]

clf_d = ExtraTreesClassifier(n_estimators=100, random_state=0)
clf_d.fit(sum_train_data, sum_train_label)
print("finish fitting d")

data = a_r_feature_machinelearning.data()
sum_train_data = data[0]
sum_train_label = data[1]

clf_a = ExtraTreesClassifier(n_estimators=100, random_state=0)
clf_a.fit(sum_train_data, sum_train_label)
print("finish fitting a")

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

n_data_1, n_test_data,n_label_1, n_test_label = train_test_split(n_data, n_label, test_size=0.1, shuffle=False)    
v_data_1, v_test_data,v_label_1, v_test_label = train_test_split(v_data, v_label, test_size=0.1, shuffle=False)
r_data_1, r_test_data,r_label_1, r_test_label = train_test_split(r_data, r_label, test_size=0.1, shuffle=False)
l_data_1, l_test_data,l_label_1, l_test_label = train_test_split(l_data, l_label, test_size=0.1, shuffle=False)
a_data_1, a_test_data,a_label_1, a_test_label = train_test_split(a_data, a_label, test_size=0.1, shuffle=False)
d_data_1, d_test_data,d_label_1, d_test_label = train_test_split(d_data, d_label, test_size=0.1, shuffle=False)
print("n",n_test_data.shape[0])
print("v",v_test_data.shape[0])
print("l",l_test_data.shape[0])
print("r",r_test_data.shape[0])
print("d",d_test_data.shape[0])
print("a",a_test_data.shape[0])

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
print("finish_")
print(sum_test_data.shape,sum_test_label.shape)

n = 0
v = 0
r = 0
l = 0
a = 0
d = 0
#filter1
predicted_nn = clf_n.predict(sum_test_data)
filter1_data = []
filter1_label = []
nn_data = []
nn_label = []
for i in range(sum_test_data.shape[0]):
    if predicted_nn[i] == "N":
        filter1_data.append(sum_test_data[i])
        filter1_label.append(sum_test_label[i])
    else:
        nn_data.append(sum_test_data[i])
        nn_label.append(sum_test_label[i])
nn_data = np.array(nn_data)
nn_label  = np.array(nn_label)
print("finish filtered n data")
for i in range(len(sum_test_label)):
    # print(predicted_nn[i],end = ",")
    if sum_test_label[i] != "N":
        sum_test_label[i] = "b"
        
a = find_elements(predicted_nn)
print(a)
print(classification_report(sum_test_label,predicted_nn))

# correct = 0
# for i in range(len(filter1_data)):
#     if filter1_label[i] == "N":
#         correct = correct +1
#         n = n+1
#     elif filter1_label[i] == "V":
#         v = v+1
#     elif filter1_label[i] == "l":
#         l = l+1
#     elif filter1_label[i] == "R":
#         r = r+1
#     elif filter1_label[i] == "/":
#         d = d+1
#     elif filter1_label[i] == "A":
#         a = a+1
    
# print("correct:",correct,"all_data",len(filter1_data))
# print("filter_n_data_accuarcy",correct/n_test_data.shape[0])
# print(n,v,l,r,d,a)



n = 0
v = 0
r = 0
l = 0
a = 0
d = 0
#filter2
sum_test_data = nn_data
sum_test_label = nn_label
predicted_nv = knn_v.predict(sum_test_data)
filter2_data = []
filter2_label = []
nv_data = []
nv_label = []
for i in range(sum_test_data.shape[0]):
    if predicted_nv[i] == "V":
        filter2_data.append(sum_test_data[i])
        filter2_label.append(sum_test_label[i])
    else:
        nv_data.append(sum_test_data[i])
        nv_label.append(sum_test_label[i])
nv_data = np.array(nv_data)
nv_label  = np.array(nv_label)
print("finish filtered v data")
for i in range(len(sum_test_label)):
    if sum_test_label[i] != "V":
        sum_test_label[i] = "b"
print(classification_report(sum_test_label,predicted_nv))

# correct = 0
# for i in range(len(filter2_data)):
#     if filter2_label[i] == "V":
#         correct = correct +1
#         v = v+1
#     elif filter2_label[i] == "N":
#         n = n+1
#     elif filter2_label[i] == "l":
#         l = l+1
#     elif filter2_label[i] == "R":
#         r = r+1
#     elif filter2_label[i] == "/":
#         d = d+1
#     elif filter2_label[i] == "A":
#         a = a+1
# print("correct:",correct,"all_data",len(filter2_data))
# print("filter_v_data_accuarcy",correct/v_test_data.shape[0])
# print(n,v,l,r,d,a)


n = 0
v = 0
r = 0
l = 0
a = 0
d = 0
#filter3
sum_test_data = nv_data
sum_test_label = nv_label
predicted_nl = knn_l.predict(sum_test_data)
filter3_data = []
filter3_label = []
nl_data = []
nl_label = []
for i in range(sum_test_data.shape[0]):
    if predicted_nl[i] == "L":
        filter3_data.append(sum_test_data[i])
        filter3_label.append(sum_test_label[i])
    else:
        nl_data.append(sum_test_data[i])
        nl_label.append(sum_test_label[i])
nl_data = np.array(nl_data)
nl_label  = np.array(nl_label)
print("finish filtered l data")
for i in range(len(sum_test_label)):
    if sum_test_label[i] != "L":
        sum_test_label[i] = "b"
print(classification_report(sum_test_label,predicted_nl))
# correct = 0
# for i in range(len(filter3_data)):
#     # print(filter3_label[i])
#     if filter3_label[i] == "L":
#         correct = correct +1
#         l = l+1
#     elif filter3_label[i] == "V":
#         v = v+1
#     elif filter3_label[i] == "N":
#         n = n+1
#     elif filter3_label[i] == "R":
#         r = r+1
#     elif filter3_label[i] == "/":
#         d = d+1
#     elif filter3_label[i] == "A":
#         a = a+1
# print("correct:",correct,"all_data",len(filter3_data))
# print("filter_l_data_accuarcy",correct/l_test_data.shape[0])
# print(n,v,l,r,d,a)



# for i in range(nl_label.shape[0]):
#     print(nl_label[i],end = ",")
#     if nn_label[0] == "A":
#         print(nn_label[0])

n = 0
v = 0
r = 0
l = 0
a = 0
d = 0
#filter4
sum_test_data = nl_data
sum_test_label = nl_label
predicted_nd = clf_d.predict(sum_test_data)
filter4_data = []
filter4_label = []
nd_data = []
nd_label = []
for i in range(sum_test_data.shape[0]):
    print(predicted_nd[i],end = ",")
    if predicted_nd[i] == "/":
        filter4_data.append(sum_test_data[i])
        filter4_label.append(sum_test_label[i])
    else:
        nd_data.append(sum_test_data[i])
        nd_label.append(sum_test_label[i])
nd_data = np.array(nd_data)
nd_label  = np.array(nd_label)
print("finish filtered d data")
for i in range(len(sum_test_label)):
    if sum_test_label[i] != "/":
        sum_test_label[i] = "b"
print(sum_test_label.shape,predicted_nd.shape)

print(classification_report(sum_test_label,predicted_nd))
# for i in range(len(sum_test_label)):
#     if sum_test_label[i] == ""
correct = 0

for i in range(len(filter4_data)):
    # print(filter4_label[i])
    if filter4_label[i] == "/":
        # print(filter4_label[i])
        correct = correct +1
        d = d+1
    elif filter4_label[i] == "V":
        v = v+1
    elif filter4_label[i] == "N":
        n = n+1
    elif filter4_label[i] == "l":
        l = l+1
    elif filter4_label[i] == "A":
        a = a+1
    elif filter4_label[i] == "R":
        r = r+1
    
# print("correct:",correct,"all_data",len(filter4_data))
# print("filter_d_data_accuarcy",correct/d_test_data.shape[0])
# print(n,v,l,r,d,a)


# n = 0
# v = 0
# r = 0
# l = 0
# a = 0
# d = 0
#filter5

sum_test_data = nd_data
sum_test_label = nd_label
print(sum_test_data.shape,sum_test_label.shape)
# result = np.savetxt("sum_test_label",sum_test_label,delimiter = ",",fmt = "%s")
# result = np.savetxt("sum_test_data",sum_test_data,delimiter = ",",fmt = "%s")
print("starting feature machine learning")

n1_data = []
n1_label = []
v1_data = []
v1_label = []
l1_data = []
l1_label = []
d1_data = []
d1_label = []
a1_data = []
a1_label = []
r1_data = []
r1_label = []


for i in range(len(sum_test_label)):
    if sum_test_label[i] == "N":
        n1_data.append(sum_test_data[i])
        n1_label.append(sum_test_label[i])
    elif sum_test_label[i] == "V":
        v1_data.append(sum_test_data[i])
        v1_label.append(sum_test_label[i])
    elif sum_test_label[i] == "L":
        l1_data.append(sum_test_data[i])
        l1_label.append(sum_test_label[i])
    elif sum_test_label[i] == "/":
        d1_data.append(sum_test_data[i])
        d1_label.append(sum_test_label[i])
    elif sum_test_label[i] == "A":
        a1_data.append(sum_test_data[i])
        a1_label.append(sum_test_label[i])
    elif sum_test_label[i] == "R":
        r1_data.append(sum_test_data[i])
        r1_label.append(sum_test_label[i])

print("n",len(n1_data))
print("v",len(v1_data))
print("l",len(l1_data))
print("d",len(d1_data))
print("a",len(a1_data))
print("r",len(r1_data))

#r1_data
r1_data = np.array(r1_data)
r1_data = r1_data.flatten()
# print(r1_data.shape)
_, rpeaks = nk.ecg_peaks(r1_data, sampling_rate=1000)
# print(rpeaks)
# print(type(rpeaks))
print("finish finding rpeak")
_, waves_peak = nk.ecg_delineate(r1_data, rpeaks, sampling_rate=1000, method="peak")
print("finish finding waves")

t_peak = np.array(waves_peak['ECG_T_Peaks'])
p_peak = np.array(waves_peak['ECG_P_Peaks'])
q_peak = np.array(waves_peak['ECG_Q_Peaks'])
s_peak = np.array(waves_peak['ECG_S_Peaks'])
onp_peak = np.array(waves_peak['ECG_P_Onsets'])
oft_peak = np.array(waves_peak['ECG_T_Offsets'])
r_peak = rpeaks['ECG_R_Peaks']
qs_distance = []
rr_distance = []
soft_distance = []
roft_distance = []
onpr_distance = []
onpoft_distance = []
first1 = 0
for i in range(r_peak.shape[0]):
    if first1 == 0:
        qs_distance.append(s_peak[i]-q_peak[i])
        soft_distance.append(oft_peak[i]-s_peak[i])
        roft_distance.append(oft_peak[i]-r_peak[i])
        onpr_distance.append(r_peak[i]-onp_peak[i])
        onpoft_distance.append(oft_peak[i]-onp_peak[i])
        onpoft_distance.append(oft_peak[i]-onp_peak[i])
        first1 = first1 + 1 
    else:
        qs_distance.append(s_peak[i]-q_peak[i])
        soft_distance.append(oft_peak[i]-s_peak[i])
        roft_distance.append(oft_peak[i]-r_peak[i])
        onpr_distance.append(r_peak[i]-onp_peak[i])
        onpoft_distance.append(oft_peak[i]-onp_peak[i])
first = 0
for i in range(r_peak.shape[0]):
    if first == 0:
        rr_distance.append(0)
        first = first + 1
    else:
        rr_distance.append(r_peak[i]-r_peak[i-1])

# print(qs_distance,rr_distance,soft_distance,roft_distance,onpr_distance,onpoft_distance)
sum_feature = []
for i in range(r_peak.shape[0]):
    if i == 0:
        sum_feature = np.hstack((qs_distance[i],soft_distance[i]))
        sum_feature = np.hstack((sum_feature,roft_distance[i]))
        sum_feature = np.hstack((sum_feature,onpr_distance[i]))
        sum_feature = np.hstack((sum_feature,onpoft_distance[i]))
        sum_feature = np.hstack((sum_feature,rr_distance[i]))
        temp_sum_feature = sum_feature
    else:
        sum_feature = np.hstack((qs_distance[i],soft_distance[i]))
        sum_feature = np.hstack((sum_feature,roft_distance[i]))
        sum_feature = np.hstack((sum_feature,onpr_distance[i]))
        sum_feature = np.hstack((sum_feature,onpoft_distance[i]))
        sum_feature = np.hstack((sum_feature,rr_distance[i])) 
        temp_sum_feature = np.vstack((temp_sum_feature,sum_feature))    
# print(temp_sum_feature)
# print(temp_sum_feature.shape[0])
# result = np.savetxt("r_feature_value",temp_sum_feature,delimiter = ",",fmt = "%s")
r_sum_feature = temp_sum_feature



"""
#n1_data
n1_data = np.array(n1_data)
n1_data = n1_data.flatten()
print(n1_data.shape)
_, rpeaks = nk.ecg_peaks(n1_data, sampling_rate=1000)
print(rpeaks)
print(type(rpeaks))
print("finish finding rpeak")
_, waves_peak = nk.ecg_delineate(n1_data, rpeaks, sampling_rate=1000, method="peak")
print("finish finding waves")

t_peak = np.array(waves_peak['ECG_T_Peaks'])
p_peak = np.array(waves_peak['ECG_P_Peaks'])
q_peak = np.array(waves_peak['ECG_Q_Peaks'])
s_peak = np.array(waves_peak['ECG_S_Peaks'])
onp_peak = np.array(waves_peak['ECG_P_Onsets'])
oft_peak = np.array(waves_peak['ECG_T_Offsets'])
r_peak = rpeaks['ECG_R_Peaks']
qs_distance = []
rr_distance = []
soft_distance = []
roft_distance = []
onpr_distance = []
onpoft_distance = []
first1 = 0
for i in range(r_peak.shape[0]):
    if first1 == 0:
        qs_distance.append(s_peak[i]-q_peak[i])
        soft_distance.append(oft_peak[i]-s_peak[i])
        roft_distance.append(oft_peak[i]-r_peak[i])
        onpr_distance.append(r_peak[i]-onp_peak[i])
        onpoft_distance.append(oft_peak[i]-onp_peak[i])
        onpoft_distance.append(oft_peak[i]-onp_peak[i])
        first1 = first1 + 1 
    else:
        qs_distance.append(s_peak[i]-q_peak[i])
        soft_distance.append(oft_peak[i]-s_peak[i])
        roft_distance.append(oft_peak[i]-r_peak[i])
        onpr_distance.append(r_peak[i]-onp_peak[i])
        onpoft_distance.append(oft_peak[i]-onp_peak[i])
first = 0
for i in range(r_peak.shape[0]):
    if first == 0:
        rr_distance.append(0)
        first = first + 1
    else:
        rr_distance.append(r_peak[i]-r_peak[i-1])

# print(qs_distance,rr_distance,soft_distance,roft_distance,onpr_distance,onpoft_distance)
sum_feature = []
for i in range(r_peak.shape[0]):
    if i == 0:
        sum_feature = np.hstack((qs_distance[i],soft_distance[i]))
        sum_feature = np.hstack((sum_feature,roft_distance[i]))
        sum_feature = np.hstack((sum_feature,onpr_distance[i]))
        sum_feature = np.hstack((sum_feature,onpoft_distance[i]))
        sum_feature = np.hstack((sum_feature,rr_distance[i]))
        temp_sum_feature = sum_feature
    else:
        sum_feature = np.hstack((qs_distance[i],soft_distance[i]))
        sum_feature = np.hstack((sum_feature,roft_distance[i]))
        sum_feature = np.hstack((sum_feature,onpr_distance[i]))
        sum_feature = np.hstack((sum_feature,onpoft_distance[i]))
        sum_feature = np.hstack((sum_feature,rr_distance[i])) 
        temp_sum_feature = np.vstack((temp_sum_feature,sum_feature))    
print(temp_sum_feature)
print(temp_sum_feature.shape[0])
result = np.savetxt("n_feature_value",temp_sum_feature,delimiter = ",",fmt = "%s")
"""


#a_data
a1_data = np.array(a1_data)
a1_data = a1_data.flatten()
print(a1_data.shape)
_, rpeaks = nk.ecg_peaks(a1_data, sampling_rate=1000)
# print(rpeaks)
# print(type(rpeaks))
print("finish finding rpeak")
_, waves_peak = nk.ecg_delineate(r1_data, rpeaks, sampling_rate=1000, method="peak")
print("finish finding waves")

t_peak = np.array(waves_peak['ECG_T_Peaks'])
p_peak = np.array(waves_peak['ECG_P_Peaks'])
q_peak = np.array(waves_peak['ECG_Q_Peaks'])
s_peak = np.array(waves_peak['ECG_S_Peaks'])
onp_peak = np.array(waves_peak['ECG_P_Onsets'])
oft_peak = np.array(waves_peak['ECG_T_Offsets'])
r_peak = rpeaks['ECG_R_Peaks']
qs_distance = []
rr_distance = []
soft_distance = []
roft_distance = []
onpr_distance = []
onpoft_distance = []
first1 = 0
for i in range(r_peak.shape[0]):
    if first1 == 0:
        qs_distance.append(s_peak[i]-q_peak[i])
        soft_distance.append(oft_peak[i]-s_peak[i])
        roft_distance.append(oft_peak[i]-r_peak[i])
        onpr_distance.append(r_peak[i]-onp_peak[i])
        onpoft_distance.append(oft_peak[i]-onp_peak[i])
        onpoft_distance.append(oft_peak[i]-onp_peak[i])
        first1 = first1 + 1 
    else:
        qs_distance.append(s_peak[i]-q_peak[i])
        soft_distance.append(oft_peak[i]-s_peak[i])
        roft_distance.append(oft_peak[i]-r_peak[i])
        onpr_distance.append(r_peak[i]-onp_peak[i])
        onpoft_distance.append(oft_peak[i]-onp_peak[i])
first = 0
for i in range(r_peak.shape[0]):
    if first == 0:
        rr_distance.append(0)
        first = first + 1
    else:
        rr_distance.append(r_peak[i]-r_peak[i-1])

# print(qs_distance,rr_distance,soft_distance,roft_distance,onpr_distance,onpoft_distance)
sum_feature = []
for i in range(r_peak.shape[0]):
    if i == 0:
        sum_feature = np.hstack((qs_distance[i],soft_distance[i]))
        sum_feature = np.hstack((sum_feature,roft_distance[i]))
        sum_feature = np.hstack((sum_feature,onpr_distance[i]))
        sum_feature = np.hstack((sum_feature,onpoft_distance[i]))
        sum_feature = np.hstack((sum_feature,rr_distance[i]))
        temp_sum_feature = sum_feature
    else:
        sum_feature = np.hstack((qs_distance[i],soft_distance[i]))
        sum_feature = np.hstack((sum_feature,roft_distance[i]))
        sum_feature = np.hstack((sum_feature,onpr_distance[i]))
        sum_feature = np.hstack((sum_feature,onpoft_distance[i]))
        sum_feature = np.hstack((sum_feature,rr_distance[i])) 
        temp_sum_feature = np.vstack((temp_sum_feature,sum_feature))    
# print(temp_sum_feature)
# print(temp_sum_feature.shape[0])
# result = np.savetxt("a_feature_value",temp_sum_feature,delimiter = ",",fmt = "%s")
a_sum_feature = temp_sum_feature


a_label = []
r_label = []
for i in range(a_sum_feature.shape[0]):
    a_label.append("A")

for i in range(r_sum_feature.shape[0]):
    r_label.append("b")


predicted = clf_a.predict(a_sum_feature)
accuracy = clf_a.score(a_sum_feature, a_label)
print(accuracy)
print(classification_report(a_label,predicted))
# for i in range(len(predicted)):
    # print(predicted[i],end=",")
predicted = clf_a.predict(r_sum_feature)
accuracy = clf_a.score(r_sum_feature, r_label)
print(accuracy)
print(classification_report(r_label,predicted))
# for i in range(len(predicted)):
    # print(predicted[i],end=",")
