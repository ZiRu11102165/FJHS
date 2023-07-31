from serial import Serial
import numpy as np
import sys
from scipy import stats
import struct
from cut_ecg import preprocess 
from sympy import diff
import os
import time
import matplotlib.pyplot as plt
from numpy import cos, sin, pi, absolute, arange
from scipy.signal import kaiserord, butter,sosfilt,lfilter, firwin, freqz
#from pylab import figure, clf, plot, xlabel, ylabel, xlim, ylim, title, grid, axes, show
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, \
    QPushButton,QLabel,QTableWidget,QTableWidgetItem
from PyQt5.QtGui import QIcon,QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QTimer
from PyQt5.QtCore import * 
import matplotlib 
matplotlib.use('Agg')
name=' '
complete=0
saveeeeeee=[]
def bytesToFloat(h1,h2,h3,h4):
    ba = bytearray()
    ba.append(h1)
    ba.append(h2)
    ba.append(h3)
    ba.append(h4)
    return struct.unpack("!f",ba)[0]
def text_save(content,filename,mode='a'):
    # Try to save a list variable in txt file.
    temp=np.array(content).shape
    file = open(filename,mode)
    for i in range(temp[0]):
        try:
            for j in range(temp[1]):
                file.write(str(content[i][j])+',')
            file.write('\n')
        except:
            file.write(str(content[i])+',')   
    file.close()

def text_read(filename):
# Try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        file = open(filename,'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    file.close()
    for i in range(len(content)):
        content[i] = content[i].split(',')
        content[i] = content[i][:len(content[i])-1]
        for j in range(len(content[i])):
            try:
                temp=int(content[i][j])
            except:
                temp=content[i][j]
                try:
                    temp=float(content[i][j])
                except:
                    temp=content[i][j]
                    try:
                        temp=complex(content[i][j])
                    except:
                        temp=content[i][j]
            content[i][j]=temp
    return content
def str_read(filename):
# Try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        file = open(filename,'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    file.close()
    for i in range(len(content)):
        content[i] = content[i][:len(content[i])-1]
    return content


class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.left = 10
        self.top = 10
        self.title = 'Read PPG and ECG'
        self.width = 1920
        self.height = 1080
        self.tim=0
        #self.setStyleSheet("background-image:url(C:/Users/leo/Desktop/123.png)")
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        #self.sub_window = SubWindow()
        self.m = PlotCanvas(self, width=5, height=5)#实例化一个画布对象
        self.m.move(0, 0)
        self.label = QLabel(self)
        self.label.move(970,330)
        self.label.resize(900,100)
        

        self.button1 = QPushButton('開始測量', self)
        self.button1.move(1450,720)
        self.button1.resize(280, 150)
        self.button1.clicked.connect(self.final_anser)
        self.button1.setStyleSheet("background-color:#2752B8 ; color: #FFFFFF ;  padding: 2px ; font: bold 45px  ; border-width: 6px ; border-radius: 10px ; border-color: #2752B8;")
        self.button1.setEnabled(False)

        self.button3 = QPushButton('初始化', self)
        self.button3.move(1100,720)
        self.button3.resize(280, 150)
        self.button3.clicked.connect(self.send_model)
        self.button3.setStyleSheet("background-color: #2B5DD1; color: #FFFFFF ; border-style: outset; padding: 2px ; font: bold 45px ; border-width: 6px ; border-radius: 10px ; border-color: #2752B8;")
        self.button5 = QPushButton('停止', self)
        self.button5.move(1620,550)
        self.button5.resize(160, 110)
        self.button5.clicked.connect(self.stop_c)
        self.button5.setStyleSheet("background-color: #2B5DD1; color: #FFFFFF ; border-style: outset; padding: 2px ; font: bold 35px ; border-width: 6px ; border-radius: 10px ; border-color: #2752B8;")
        self.button5.setVisible(False)
        #self.button3.setVisible(False)
        self.show()


    def send_model(self):
        self.m.build_l()
        self.button1.setStyleSheet("background-color: #2B5DD1; color: #FFFFFF ; border-style: outset; padding: 2px ; font: bold 45px ; border-width: 6px ; border-radius: 10px ; border-color: #2752B8;")
        self.button1.setEnabled(True)
        # self.label_hint.setText("請貼上電極，配戴完畢後按下開始測量")

    def stop_c(self):
        self.button5.setVisible(False)
        # self.label_hint.setText("測量完畢!")
        self.m.cheat=4
        
    def final_anser(self):
        self.m.plot2()      

        self.show()
        self.timer1 = QTimer(self)       
        self.timer1.timeout.connect(self.put_data)
        self.timer1.start(0.5)

    def put_data(self):
        if self.m.cheat==1:
            peo,bo=self.m.get_data()

            self.m.cheat=3
            self.button5.setVisible(True)
            # print('ok')
            self.m.plot3() 
            #self.timer1.stop()
            #self.m.cheat=0
        elif self.m.cheat==5:
            peo,bo=self.m.get_data()

            self.m.cheat=0
            #self.button5.setVisible(True)
            #self.m.plot3() 
            self.timer1.stop()
            #self.m.cheat=0


class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=20, height=20, dpi=190):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.subplots_adjust(0.1, 0.1, 0.9, 0.9)
        fig.patch.set_facecolor((0.1, 0.2, 0.5,0.3))
        self.axes = fig.add_subplot(211)
        self.axes.patch.set_facecolor((0.1, 0.2, 0.5,0.05))
        self.axes1 = fig.add_subplot(212)
        self.axes1.patch.set_facecolor((0.1, 0.2, 0.5,0.05))
        self.ser_arc = Serial('COM12',38400,bytesize=8)
        self.hhr = 80
        self.sbp = 70
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        self.first=0
        self.count=0
        self.data=[]
        self.data2=[]
        self.cheat=0
        self.ss=0
        self.test_data=[]
        self.long_data=[]
        self.ecg_save=[]
        self.ppg_save=[]
        self.data_name = ''
        self.start1 = time.time()
        self.end1 = time.time()
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.init_plot()#打开App时可以初始化图片
        #self.plot()
    def get_data(self):
        return self.sp[self.ans][1],self.blood
    def plot3(self):
        # print('ok1')
        self.start1 = time.time()
        
        self.timer = QTimer(self)
        
        self.timer.timeout.connect(self.update_figure2)
        self.timer.start(1)

    def update_figure2(self):
        w=0
        # print('open')
        if self.first==0:
            # print('time3')
            self.end1 = time.time()
            # print(self.end1 - self.start1)
            self.start1 = time.time()
            self.data=[]
            self.data2=[]
            self.first=1
            self.count=0
        if self.count<416:
            while True:
                while w!=960:
                    data_raw = self.ser_arc.read(size=1)
                    # print(data_raw)
                    #print(self.count)
                    if data_raw==b'c':
                        # print('ok')
                        data_raw = self.ser_arc.read(size=4)
                        for i in range(1):
                            ba = bytearray()
                            ba.append(data_raw[i*3+3])
                            ba.append(data_raw[i*3+2])
                            ba.append(data_raw[i*3+1])
                            ba.append(data_raw[i*3+0])
                            # print(struct.unpack("!f",ba)[0])
                            self.data.append(struct.unpack("!f",ba)[0])
                            self.long_data.append(struct.unpack("!f",ba)[0])
                        data_raw = self.ser_arc.read(size=4)
                        for i in range(1):
                            ba = bytearray()
                            ba.append(data_raw[i*3+3])
                            ba.append(data_raw[i*3+2])
                            ba.append(data_raw[i*3+1])
                            ba.append(data_raw[i*3+0])
                        
                            # if struct.unpack("!f",ba)[0]==0:
                            #     self.data2.append(self.data2[len(self.data2)-1])
                            # else:
                            # print(ba)
                            self.data2.append(struct.unpack("!f",ba)[0])
                            
                        w=w+1
                        self.count=self.count+1
                        #print(self.count)
                        # print(self.count)
                        # print(w)
                    if w==26 or self.count==416:
                        break
                if w==26 or self.count==416: 

                    if len(self.ecg_save)<416:
                        for i in range(len(self.data)):
                            self.ecg_save.append(self.data[i])
                    else:
                        
                        self.ecg_save[0:391]=self.ecg_save[25:416]
                        self.ecg_save[391:416]=self.data[0:25]
                    if len(self.ppg_save)<416:
                        for i in range(len(self.data2)):
                            self.ppg_save.append(self.data2[i])
                    else:
                        sos = butter(7, 30, fs=400, output='sos')
                        # filtered = sosfilt(sos, ppg_save)
                        self.ppg_save[0:391]=self.ppg_save[25:416]
                        self.ppg_save[391:416]=self.data2[0:25]
                    self.data=[]
                    self.data2=[]
                    self.axes.cla()
                    self.axes1.cla()
                    self.axes.plot(self.ecg_save)
                    self.axes1.plot(self.ppg_save)
                    self.draw()
                    break
        else:

            self.timer.stop()

            self.res1()
            self.first=0
            self.count=0


    def plot2(self):
        self.timer = QTimer(self)
        
        self.timer.timeout.connect(self.update_figure1)
        self.timer.start(0.5)

    def update_figure1(self):
        w=0
        if self.first==0:
            start='g'
            # print(start.encode())
            self.ser_arc.write(start.encode())
            self.data=[]
            self.data2=[]
            self.first=1
        if self.count<960:
            while True:
                while w!=960:
                    data_raw = self.ser_arc.read(size=1)
                    # print(data_raw)
                    # print(self.count)
                    if data_raw==b'c':
                        # print('ok')
                        data_raw = self.ser_arc.read(size=4)
                        for i in range(1):
                            ba = bytearray()
                            ba.append(data_raw[i*3+3])
                            ba.append(data_raw[i*3+2])
                            ba.append(data_raw[i*3+1])
                            ba.append(data_raw[i*3+0])
                            # print(ba)
                            # print(struct.unpack("!f",ba)[0])
                            self.data.append(struct.unpack("!f",ba)[0])
                        data_raw = self.ser_arc.read(size=4)
                        for i in range(1):
                            ba = bytearray()
                            ba.append(data_raw[i*3+3])
                            ba.append(data_raw[i*3+2])
                            ba.append(data_raw[i*3+1])
                            ba.append(data_raw[i*3+0])
                            # print(ba)
                            # print(struct.unpack("!f",ba)[0])
                            self.data2.append(struct.unpack("!f",ba)[0])
                        w=w+1
                        self.count=self.count+1
                    if w==20:
                        break
                if w==20:
                    break
        else:
            self.timer.stop()
            self.res()
            self.first=0
            self.count=0
        if len(self.data)>200:
            
            # np.savetxt('look.txt',self.data)
            self.axes.cla()
            self.axes.plot(self.data[200:])
            self.axes1.cla()
            self.axes1.plot(self.data2[200:])
            self.draw()
            
    def plot1(self):
        self.timer = QTimer(self)
        
        self.timer.timeout.connect(self.update_figure)
        self.timer.start(0.5)
    def update_figure(self):
        w=0
        if self.first==0:
            start='s'
            # print(start.encode())
            self.ser_arc.write(start.encode())
            self.data=[]
            self.data2=[]
            self.first=1
        if self.count<960:
            while True:
                while w!=960:
                    data_raw = self.ser_arc.read(size=1)
                    # print(data_raw)
                    # print(self.count)
                    if data_raw==b'c':
                        # print('ok')
                        data_raw = self.ser_arc.read(size=4)
                        for i in range(1):
                            ba = bytearray()
                            ba.append(data_raw[i*3+3])
                            ba.append(data_raw[i*3+2])
                            ba.append(data_raw[i*3+1])
                            ba.append(data_raw[i*3+0])
                            # print(struct.unpack("!f",ba)[0])
                            self.data.append(struct.unpack("!f",ba)[0])
                        data_raw = self.ser_arc.read(size=4)
                        for i in range(1):
                            ba = bytearray()
                            ba.append(data_raw[i*3+3])
                            ba.append(data_raw[i*3+2])
                            ba.append(data_raw[i*3+1])
                            ba.append(data_raw[i*3+0])
                            # print(struct.unpack("!f",ba)[0])
                            self.data2.append(struct.unpack("!f",ba)[0])
                        w=w+1
                        self.count=self.count+1
                    if w==20:
                        break
                if w==20:
                    break
        else:
            self.timer.stop()
            self.save_data()
            self.first=0
            self.count=0

        self.axes.cla()
        self.axes.plot(self.data[200:])
        self.axes1.cla()
        self.axes1.plot(self.data2[200:])
        self.draw()

    def save_data(self):
        # print("go")
        a=[]
        b=[]
        c=[]
        d=[]
        start='s'
        w=0
        while True:
            while w!=960:
                data_raw = self.ser_arc.read(size=1)
                # print(data_raw)
                if data_raw==b'c':
                    # print('ok')
                    # print(w)
                    data_raw = self.ser_arc.read(size=4)
                    for i in range(1):
                        ba = bytearray()
                        ba.append(data_raw[i*3+3])
                        ba.append(data_raw[i*3+2])
                        ba.append(data_raw[i*3+1])
                        ba.append(data_raw[i*3+0])
                        # print(struct.unpack("!f",ba)[0])
                        if i==0:
                            a.append(struct.unpack("!f",ba)[0])
                        elif i==1:
                            b.append(struct.unpack("!f",ba)[0])
                        elif i==2:
                            c.append(struct.unpack("!f",ba)[0])
                        elif i==3:
                            d.append(struct.unpack("!f",ba)[0])
                    w=w+1
            if w==960:
                break
        w=0
        while True:
            while w!=960:
                data_raw = self.ser_arc.read(size=1)
                # print(data_raw)
                if data_raw==b'c':
                    # print('ok')
                    # print(w)
                    data_raw = self.ser_arc.read(size=4)
                    for i in range(1):
                        ba = bytearray()
                        ba.append(data_raw[i*3+3])
                        ba.append(data_raw[i*3+2])
                        ba.append(data_raw[i*3+1])
                        ba.append(data_raw[i*3+0])
                        # print(struct.unpack("!f",ba)[0])
                        if i==0:
                            b.append(struct.unpack("!f",ba)[0])
                        elif i==1:
                            b.append(struct.unpack("!f",ba)[0])
                        elif i==2:
                            c.append(struct.unpack("!f",ba)[0])
                        elif i==3:
                            d.append(struct.unpack("!f",ba)[0])
                    w=w+1
            if w==960:
                break
        f=0
        while True:
            data_raw = self.ser_arc.read(size=1)
            if data_raw==b'd':
                data_raw = self.ser_arc.read(size=2)
                print("done")
                f=1
            if data_raw==b'e':
                data_raw = self.ser_arc.read(size=2)
                f=1
            if f==1:
                break

        font=QFont()
        font.setFamily("Arial")
        font.setPointSize(50)

        self.axes.cla()
        self.axes.plot(a[200:])
        self.axes1.cla()
        self.axes1.plot(b[200:])
        self.draw()
        # print("final")
    def build_l(self):
        start='r'
        self.ser_arc.write(start.encode())
        di=os.listdir()
        first=0
        all_target=np.array([])
        all_data=np.array([])
        all_target=0
        sp=text_read("name"+".txt")
        sp=np.array(sp)
        n=int(sp[sp.shape[0]-1][0])+1
        n=str(n)
        start='p'
        self.ser_arc.write(start.encode())
        #print(n.encode())
        self.ser_arc.write(struct.pack('c',n.encode()))
        n=int(n)
        people_num=n
        for i in range(n):
            for na in di:

                if na[0:2]==('0'+str(i)):
                    # print(na[0:4])
                    data=preprocess(na[0:4],0)
                    
                    if first==0:
                        first=1
                        all_data=data
                        for w in range(len(data)-1):
                            all_target=np.vstack((all_target,i))
                    else:
                        all_data=np.vstack((all_data,data))
                        if len(data)==100:
                            all_target=np.vstack((all_target,i))
                        else:
                            for w in range(len(data)):
                                all_target=np.vstack((all_target,i))

        sklda= LinearDiscriminantAnalysis()
        data_2 = sklda.fit_transform(all_data, all_target)
        b=sklda.scalings_
        mean=np.zeros((people_num,people_num-1))
        mean_save=[]
        for i in range(people_num-1):
            mean_save.append(0)
        a=all_data
        c=np.dot(a,b)
        target=0
        count=0
        flag=0
        #print(c)
        for i in range(all_target.shape[0]):
        
            if all_target[i,0]!=target:
                for w in range(people_num-1):
                    mean_save[w]=mean_save[w]/count
                if flag!=1:
                    for w in range(people_num-1):
                        mean[target,w]=mean_save[w]

                else :
                    

                    for w in range(people_num-1):
                        mean[target,w]=mean_save[w]
                for w in range(people_num-1):
                    mean_save[w]=c[i,w]
                target=all_target[i,0]
                flag=1
                count=0
            else: 
                for w in range(people_num-1):
                    # print(mean_save)
                    mean_save[w]=mean_save[w]+c[i,w]
            
            count=count+1
        for i in range(people_num-1):
            mean_save[i]=mean_save[i]/count
            mean[target,i]=mean_save[i] 
        # print("mean")
        # print(mean)
        # #print("b")
        # #print(b)
        start='t'
        self.ser_arc.write(start.encode())
        for i in range(100):
            for j in range(n-1):
                self.ser_arc.write(struct.pack('f',b[i][j]))
        start='m'
        self.ser_arc.write(start.encode())
        for i in range(n):
            for j in range(n-1):
                self.ser_arc.write(struct.pack('f',mean[i][j]))
        people_num=int(people_num)
        start='b'
        self.ser_arc.write(start.encode())
        for peo in range(people_num):
            people=str(peo)
            X_s=text_read("sp_0"+people+".txt")
            s_blood=text_read("blood_s_0"+people+".txt")
            X_s=np.array(X_s)
            s_blood=np.array(s_blood)
            ad=1
            for q in range(X_s.shape[0]-1):
                ad=np.vstack((ad,1))
            X_s=np.hstack((X_s,ad))
            re1=np.dot(np.dot(np.linalg.inv(np.dot(X_s.T,X_s)),X_s.T),s_blood).T
            self.ser_arc.write(struct.pack('f',re1[0][0]))
            self.ser_arc.write(struct.pack('f',re1[0][1]))
        #     print(re1)
        # #print(data_2.shape)

    def res1(self):
        
        f=0
        ff=0
        identity=0
        self.blood=0
        self.ans1=0

        while True:
            data_raw = self.ser_arc.read(size=1)
            # print(data_raw)
            if data_raw==b'c':
                # print('ok')
                data_raw = self.ser_arc.read(size=1)
                ptt=int.from_bytes(data_raw,byteorder='big')
                hrr = self.ser_arc.read(size=1)
                hrr=int.from_bytes(hrr,byteorder='big')

                break

        while True:
            data_raw = self.ser_arc.read(size=1)
            # print(data_raw)
            if data_raw==b'e':
                break
        X_s=text_read("sp_01.txt")
        s_blood=text_read("blood_s_01.txt")
        X_s = np.array(X_s)
        s_blood = np.array(s_blood)
        # print(X_s)
        # print(s_blood)
        ad=1
        for q in range(X_s.shape[0]-1):
            ad=np.vstack((ad,1))
        X_s=np.hstack((X_s,ad))
        re1=np.dot(np.dot(np.linalg.inv(np.dot(X_s.T,X_s)),X_s.T),s_blood).T

        self.blood=int(re1[0][0]*ptt+re1[0][1])
        self.hhr = 60/(hrr/200)
        self.sbp=ptt*0.4+40
        #ans=int(ans,16)
        self.sp=text_read("name"+".txt")
        self.sp=np.array(self.sp)
        if self.cheat==3:
            self.cheat=1
            # start='g'
            # print(start.encode())
            # self.ser_arc.write(start.encode())
        elif self.cheat==4:
            # print('w')
            self.cheat=5
            start='w'
            self.ser_arc.write(start.encode())
            while True:
                data_raw = self.ser_arc.read(size=1)
                # print(data_raw)
                if data_raw==b'u':
                    data_raw = self.ser_arc.read(size=1)
                    if data_raw==b'w':
                        break
            # self.ser_arc.close()
            # time.sleep(1)
            # self.ser_arc = Serial('COM8',38400,bytesize=8)
        return self.sp[self.ans][1],self.blood

    def res(self):
        a=[]
        b=[]
        c=[]
        d=[]
        self.ecg_save=[]
        self.ppg_save=[]
        w=0
        f=0
        ff=0
        identity=0
        self.blood=0
        self.ans=0
        while True:
            data_raw = self.ser_arc.read(size=1)
            # print(data_raw)
            if data_raw==b'r':
                data_raw = self.ser_arc.read(size=1)
                self.ans=data_raw
                # print(data_raw)
            if data_raw==b'c':
                # print('ok')
                data_raw = self.ser_arc.read(size=4)
                ba = bytearray()
                ba.append(data_raw[3])
                ba.append(data_raw[2])
                ba.append(data_raw[1])
                ba.append(data_raw[0])
                # print(struct.unpack("!f",ba)[0])
                self.blood=struct.unpack("!f",ba)[0]
                break
        while True:
            data_raw = self.ser_arc.read(size=1)
            # print(data_raw)
            if data_raw==b'c':
                # print('ok')
                data_raw = self.ser_arc.read(size=4)
                ba = bytearray()
                ba.append(data_raw[3])
                ba.append(data_raw[2])
                ba.append(data_raw[1])
                ba.append(data_raw[0])
                # print(struct.unpack("!f",ba)[0])
                ptt=struct.unpack("!f",ba)[0]
                break

        while True:
            data_raw = self.ser_arc.read(size=1)
            # print(data_raw)
            if data_raw==b'e':
                # print("done e")
                break
        self.axes.cla()
        self.axes.plot(a[200:])
        self.axes1.cla()
        self.axes1.plot(b[200:])
        self.draw()

        self.blood=130#*(-0.71)+159.13
        #ans=int(ans,16)
        self.sp=text_read("name"+".txt")
        self.sp=np.array(self.sp)
        #print(self.ans)
        if self.ans!=0:
            self.ans=int.from_bytes(self.ans, byteorder='big')
        self.cheat=1;
        return self.sp[self.ans][1],self.blood
    
    def plot(self):
        self.ser_arc = Serial('COM12',38400,bytesize=8)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_figure)
        self.timer.start(0.5)

    def init_plot(self):
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        y = [23, 21, 32, 13, 3, 132, 13, 3, 1]
        self.axes.plot(x, y,'r')
        self.axes1.plot(x, y,'r')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    #app.setStyleSheet(stylesheet)
    ex = App()
    sys.exit(app.exec_())

