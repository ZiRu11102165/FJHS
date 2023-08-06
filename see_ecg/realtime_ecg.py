from serial import Serial
import struct
import matplotlib.pyplot as plt
import os
import numpy as np
import keyboard  

w=0
ser_arc = Serial('COM12',38400,bytesize=8)#Check the COM port in your computer and chage
count=0
data=[]
data2=[]

import matplotlib.pyplot as plt
import numpy as np

# 動態畫圖用
plt.ion()

# 創建一個空的折線圖
fig, ax = plt.subplots()
line, = ax.plot([], [])  # 空的折線圖，將在更新時填充數據
ax.set_xlim(0, 250)  # X軸範圍
ax.set_ylim(200, 650)  # Y軸範圍   可能要改成自動
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.set_title('Real-time Data')
plt.grid()

data = []

#####################################Make sure the connect
while True:
    data_raw = ser_arc.read(size=1)
    # print(data_raw)
    # print(count)
    if data_raw==b'c':
        print('ok')
        data_raw = ser_arc.read(size=4)
        for i in range(1):                  
            ba = bytearray()
            ba.append(data_raw[i*3+3])      
            ba.append(data_raw[i*3+2])      
            ba.append(data_raw[i*3+1])
            ba.append(data_raw[i*3+0])
            print(struct.unpack("!f",ba)[0])    
            data.append(struct.unpack("!f",ba)[0])     
        # 只保留最新的100筆資料
        data_e = data[-250:]
        
        # 更新折線圖的數據
        xdata = np.arange(len(data_e))
        line.set_data(xdata, data_e)  
        plt.draw()  
        plt.pause(0.05)  
        plt.show()
        data_raw = ser_arc.read(size=4)
    if keyboard.is_pressed('esc'):
        break
#####################################save file
di=os.listdir()
maximum = 0
for name in di:
    if name[0]=='0':
        if maximum<int(name[0:4]):
            maximum = int(name[0:4])
maximum=maximum+1
if maximum>99:
    np.savetxt('0'+str(maximum)+'.txt',data)
    # np.savetxt('0'+str(maximum)+'_p.txt',data2)
elif maximum>9:
    np.savetxt('00'+str(maximum)+'.txt',data)
    # np.savetxt('00'+str(maximum)+'_p.txt',data2)
else:
    np.savetxt('000'+str(maximum)+'.txt',data)
    # np.savetxt('000'+str(maximum)+'_p.txt',data2)

print('save done')

