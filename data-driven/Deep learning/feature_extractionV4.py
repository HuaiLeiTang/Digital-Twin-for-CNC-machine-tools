'''
版本：V4

作者：骆伟超

说明：首先计算幅域分析,时域分析,和频域分析,进行特征的提取. 看一下信号的特征,是否特征提取合理可行.
'''
import os
import pandas as pd
import numpy as np

##========================幅域分析=================================
##计算幅域的平均值
def averageValue(a):
	return sum(a)/len(a)

##计算幅域的有效值
def Xrms(a):
	b=[i*i for i in a]
	return (sum(b)/len(b))**0.5


# 峰值 用max(list),最小值用min(list)


##计算幅域的方根幅值
def Xr(a):
	b=[abs(i)**0.5 for i in a]
	return (sum(b)/len(b))**2


##计算幅域的歪度,反应信号的不对中特性,歪度是对幅值的3次方 进行数学处理
def waidu(a):
	b=[i**3 for i in a]
	return sum(b)/len(b)

##计算幅域的峭度, 峭度值对大幅值非常敏感，当大幅值增多时峭度值迅速增加，这非常有利于检测含有脉冲冲击性的故障信号。
def qiaodu(a):
	b=[i**4 for i in a]
	return sum(b)/len(b)
##========================幅域分析=================================


Clmnames=['F_X', 'F_Y', 'F_Z', 'V_X', 'V_Y', 'V_Z', 'S_E']
timeFeatures=["Xrms", "max"]
Frefeatures = ["freq1", "value1", "freq2", "value2", "freq3", "value3"]
FrefeaturesLen = 3


timeClm=[]
for x in Clmnames:
	for y in timeFeatures:
		timeClm.append(x+'_'+y)


freaClm = []
for x in Clmnames[:6]:
	for y in Frefeatures:
		freaClm.append(x + '_' + y)

dataFrame = pd.DataFrame(columns=timeClm+freaClm)  #创建一个空的dataframe
print(dataFrame)

##进行文件排列进行顺序处理
files=os.listdir('.')
files.sort()

for fileName in files:
	if not fileName.endswith('.csv'):
		continue
		# skip non-csv files
	print('      ### Processing   ' + fileName + '###      ')
	file = pd.read_csv(fileName, names=Clmnames,header=0)
	
	effective_data=file[40000:90000]

	trial_features=[] ##用来存储每次试验的所有特征值
	for i in range(7):
		column = file.iloc[:, i]
		##print(averageValue(column),Xrms(column),max(column),Xr(column),waidu(column),qiaodu(column))
		trial_features=trial_features+[Xrms(column),max(column)] ##列表拼接

	# 计算主频的幅值
	t = [x / 50000.0 for x in range(len(effective_data))]  #取一秒钟的数据
	#pl.plot(t,effective_data.F_X)
	#pl.show()
	
	N = len(t)  # 采样点数 50000个
	fs = 50000  # 采样频率
	df = fs / (N - 1)  # 分辨率
	f = [df * n for n in range(0, N)]  # 构建频率数组
	
	for i in range(6):
		column = effective_data.iloc[:, i]
		
		##使用下面的代码进行快速傅里叶变换并对结果模值进行缩放：
		Y = np.fft.fft(column) * 2 / N  #*2/N 反映了FFT变换的结果与实际信号幅值之间的关系
		absY = [np.abs(x) for x in Y]  #求傅里叶变换结果的模
		absY = absY[1:25000]  ##去掉0频分量
		
		Freqdict = {}
		for i in range(len(absY)):
			Freqdict[i] = absY[i]
		
		FreSorted = sorted(Freqdict.items(), key=lambda item: item[1], reverse=True)
		FreSortedEffective = FreSorted[:30]
		#print(FreSortedEffective)
		dropList = []
		for i in range(1, len(FreSortedEffective)):
			for j in range(i):
				if FreSortedEffective[j][0] - 10 < FreSortedEffective[i][0] < FreSortedEffective[j][0] + 10:
					dropList.append(FreSortedEffective[i])
		#print(FreSortedEffective[:i])
		#print(dropList)
		
		for item in dropList:
			if item in FreSortedEffective:
				FreSortedEffective.remove(item)
		#print(FreSortedEffective[:5])
		while len(FreSortedEffective) < FrefeaturesLen:
			FreSortedEffective.append((0, 0))
		for item in FreSortedEffective[:FrefeaturesLen]:
			trial_features.append(item[0])
			trial_features.append(item[1])
	#print(trial_features)
	dataFrame.loc[dataFrame.shape[0]] = trial_features
dataFrame.to_csv("FeatureTest.csv")


