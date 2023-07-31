import numpy as np
from sklearn.model_selection import train_test_split

def data():
    r_feature = []
    a_feature = []
    r_label = []
    a_label = []

    r_data  = np.loadtxt("C:/Users/ASUS/OneDrive/桌面/python/course/mit-bih-arrhythmia-database-1.0.0/r_feature_value",delimiter=",")
    a_data = np.loadtxt("C:/Users/ASUS/OneDrive/桌面/python/course/mit-bih-arrhythmia-database-1.0.0/a_feature_value",delimiter=",")
    for i in range(len(r_data)):
        # r_feature.append(r_data[i][0])
        r_label.append("b")
    for i in range(len(a_data)):
        # a_feature.append(a_data[i][0])
        a_label.append("A")
    a_feature = a_data
    r_feature = r_data
    r_feature, r_test_data,r_label, r_test_label = train_test_split(r_feature, r_label, test_size=0.1, shuffle=False)
    a_feature, a_test_data,a_label, a_test_label = train_test_split(a_feature, a_label, test_size=0.1, shuffle=False)

    print(len(r_feature),len(r_label))
    print(len(a_feature),len(a_label))
    # print(r_label)
    # print(a_label)
    sum_data = np.vstack((r_feature,a_feature))
    print(sum_data.shape)
    sum_label = np.hstack((r_label,a_label))
    print(sum_label.shape)

    sum_train_data,sum_test_data,sum_train_label,sum_test_label = train_test_split(sum_data,sum_label,random_state = 42,train_size = 0.8)
    print(sum_train_data.shape,sum_test_data.shape,sum_train_label.shape,sum_test_label.shape)
    return sum_train_data,sum_train_label

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_classification

# clf_a_r = ExtraTreesClassifier(n_estimators=100, random_state=0)
# clf_a_r.fit(sum_train_data, sum_train_label)


# predicted = clf_a_r.predict(sum_test_data)
# accuracy = clf_a_r.score(sum_test_data, sum_test_label)
# print(accuracy)
# print(r_feature.shape,a_feature.shape)
"""
print(len(r_feature))
print(len(a_feature))
normal_train_data,normal_test_data,normal_train_label,normal_test_label = train_test_split(a_feature,a_label,random_state = 42,train_size = 0.8)
abnormal_train_data,abnormal_test_data,abnormal_train_label,abnormal_test_label = train_test_split(a_feature,a_label,random_state = 42,train_size = 0.8)

sum_train_label = np.vstack((normal_train_label,abnormal_train_label))
sum_test_label = np.vstack((normal_test_label,abnormal_test_label))
sum_train_data = np.vstack((normal_train_data,abnormal_train_data))
sum_test_data = np.vstack((normal_test_data,abnormal_test_data))
print(sum_train_data.shape,sum_test_data.shape,sum_train_label.shape,sum_test_label.shape)

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_classification

clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
clf.fit(sum_train_data, sum_train_label)
predicted = clf.predict(sum_test_data_1)
accuracy = clf.score(sum_test_data_1, sum_test_label_1)
# print(r_feature.shape,a_feature.shape)



"""

def data():
    r_feature = []
    a_feature = []
    r_label = []
    a_label = []

    r_data  = np.loadtxt("C:/Users/ASUS/OneDrive/桌面/python/course/mit-bih-arrhythmia-database-1.0.0/r_feature_value",delimiter=",")
    a_data = np.loadtxt("C:/Users/ASUS/OneDrive/桌面/python/course/mit-bih-arrhythmia-database-1.0.0/a_feature_value",delimiter=",")

    for i in range(len(r_data)):
    # r_feature.append(r_data[i][0])
        r_label.append("b")
    for i in range(len(a_data)):
    # a_feature.append(a_data[i][0])
        a_label.append("A")
    a_feature = a_data
    r_feature = r_data
    r_feature, r_test_data,r_label, r_test_label = train_test_split(r_feature, r_label, test_size=0.1, shuffle=False)
    a_feature, a_test_data,a_label, a_test_label = train_test_split(a_feature, a_label, test_size=0.1, shuffle=False)

    print(len(r_feature),len(r_label))
    print(len(a_feature),len(a_label))
    # print(r_label)
    # print(a_label)
    sum_data = np.vstack((r_feature,a_feature))
    print(sum_data.shape)
    sum_label = np.hstack((r_label,a_label))
    print(sum_label.shape)

    sum_train_data,sum_test_data,sum_train_label,sum_test_label = train_test_split(sum_data,sum_label,random_state = 42,train_size = 0.8)
    print(sum_train_data.shape,sum_test_data.shape,sum_train_label.shape,sum_test_label.shape)

    return sum_test_data,sum_test_label,sum_train_data,sum_train_label