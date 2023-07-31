import wfdb
import numpy as np
import matplotlib.pyplot as plt

def getDataSet(number, X_data,label,label_point):

    record = wfdb.rdrecord('mit-bih-arrhythmia-database-1.0.0/' + number, channel_names=['MLII'])
    data = record.p_signal.flatten()
    annotation = wfdb.rdann('mit-bih-arrhythmia-database-1.0.0/' + number, 'atr')
    Rlocation = annotation.sample 
    Rclass = annotation.symbol
    X_data.append(data)
    label.append(Rclass)
    label_point.append(Rlocation)
    return 

def loadData():
    numberSet = ['100', '101', '103', '105', '106', '107', '108', '109', '111', '112', '113', '114', '115',
                 '116', '117', '119', '121', '122', '123', '124', '200', '201', '202', '203', '205', '208',
                 '210', '212', '213', '214', '215', '217', '219', '220', '221', '222', '223', '228', '230',
                 '231', '232', '233', '234']
    dataSet = []
    label = []
    label_point = []
    for n in numberSet:
        getDataSet(n, dataSet,label,label_point)
    return dataSet,label,label_point

def main():
    dataSet,label,label_point = loadData()
    dataSet = np.array(dataSet)
    print(dataSet.shape)
    print(label_point[0][0])
    plt.plot(dataSet[0,label_point[0][1] - 50 : label_point[0][1] + 50])
    plt.show()
    print("data ok!!!")

if __name__ == '__main__':
    main()