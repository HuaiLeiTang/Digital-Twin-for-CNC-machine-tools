import keras
import keras.backend as K
from keras.layers.core import Activation
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, LSTM

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing

# Setting seed for reproducibility
np.random.seed(1234)
PYTHONHASHSEED = 0

# define path to save model
model_path = './regression_model.h5'

##################################
# Data Ingestion
##################################

# read training data - It is the tool run-to-failure data.
train_df = pd.read_csv('./FeatureA.csv', sep=",", header=0)
train_df.drop(train_df.columns[0], axis=1, inplace=True) ##去除time列
train_dfB = pd.read_csv('./FeatureB.csv', sep=",", header=0)
train_dfB.drop(train_df.columns[0], axis=1, inplace=True) ##去除time列
# 读取磨损数据
mosun_df = pd.read_csv('./Train_A_wear.csv', sep=",", header=None) ##读取真实数据，只有一列，都是真实的剩余寿命
mosun_df.columns=['flute1','flute2','flute3','max']
mosun_dfB = pd.read_csv('./Train_B_wear.csv', sep=",", header=None) ##读取真实数据，只有一列，都是真实的剩余寿命
mosun_dfB.columns=['flute1','flute2','flute3','max']

# read test data - It is the aircraft engine operating data without failure events recorded.
test_df = pd.read_csv('./FeatureTest.csv', sep=",", header=0)
test_df.drop(test_df.columns[0], axis=1, inplace=True) ##去除time列

# read ground truth data - It contains the information of true remaining cycles for each engine in the testing data.
truth_df = pd.read_csv('./Test_wear.csv', sep=",", header=None) ##读取真实数据，只有一列，都是真实的剩余寿命
truth_df.columns=['flute1','flute2','flute3','max']


##################################
# Prepare training data
##################################
train_df ['mosun']= mosun_df['max']
train_dfB ['mosun']= mosun_dfB['max']
#print(train_df)


# MinMax normalization (from 0 to 1)
## 找出与这些不同的列，也就是找出要归一化的列
cols_normalize = train_df.columns.difference(['mosun'])
print(cols_normalize)

min_max_scaler = preprocessing.MinMaxScaler()   ##最大最小值标准化，将数据标准到[0,1]之间。
norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(train_df[cols_normalize]),
                             columns=cols_normalize,
                             index=train_df.index)
join_df = train_df[train_df.columns.difference(cols_normalize)].join(norm_train_df) ##归一化和不归一化的合并到一起。
train_df = join_df.reindex(columns = train_df.columns) ##按照原来的列顺序重新排列一遍。
train_df.to_csv('./PredictiveManteinanceTraining.csv', encoding='utf-8',index = None) ##保存成一个csv文件


##################################
# Prepare test data
##################################
test_df ['mosun']= truth_df['max']
# MinMax normalization (from 0 to 1)
##test_df['mosun_norm'] = test_df['mosun']
norm_test_df = pd.DataFrame(min_max_scaler.transform(test_df[cols_normalize]),
                            columns=cols_normalize,
                            index=test_df.index)
test_join_df = test_df[test_df.columns.difference(cols_normalize)].join(norm_test_df)
test_df = test_join_df.reindex(columns=test_df.columns)
test_df = test_df.reset_index(drop=True)
test_df.to_csv('./PredictiveManteinanceTesting.csv', encoding='utf-8',index = None) ##保存成一个csv文件
#print(test_df)

# pick a large window size of 20 cycles
sequence_length = 20


# function to reshape features into (samples, time steps, features)
def gen_sequence(id_df, seq_length, seq_cols):
    """ Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length. An alternative would be to pad sequences so that
    we can use shorter ones """
    #这个函数写的很简短，如果是我来写的话可能会用for循环了
    # for one id I put all the rows in a single matrix
    data_matrix = id_df[seq_cols].values ##获取传入的矩阵按seq_cols布局的数值
    num_elements = data_matrix.shape[0] ##返回传入矩阵的行数
    # Iterate over two lists in parallel.
    # For example id1 have 192 rows and sequence_length is equal to 50
    # so zip iterate over two following list of numbers (0,112),(50,192)
    # 0 50 -> from row 0 to row 50
    # 1 51 -> from row 1 to row 51
    # 2 52 -> from row 2 to row 52
    # ...
    # 141 191 -> from row 141 to 191
    ##通过zip将对应的如[0:141][50:191]打包成[(0，50),.....，（141,191）]这样的序列
    for start, stop in zip(range(0, num_elements - seq_length), range(seq_length, num_elements)):
        yield data_matrix[start:stop, :] ##将对应行数的所有列都返回出来


# TODO for debug
# val is a list of 192 - 50 = 142 bi-dimensional array (50 rows x 25 columns)
# 这里应该是为了产生循环的序列，为了是作为LSTM的输入信号。

val = gen_sequence(train_df, sequence_length, cols_normalize)

# generate sequences and convert to numpy array
#seq_array = np.concatenate(val).astype(np.float32)
seq_array = np.array(list(val))
print("训练数据的格式")
print(seq_array.shape)


# function to generate labels
def gen_labels(id_df, seq_length, label):
    """ Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length. An alternative would be to pad sequences so that
    we can use shorter ones """
    # For one id I put all the labels in a single matrix.
    # For example:
    # [[1]
    # [4]
    # [1]
    # [5]
    # [9]
    # ...
    # [200]]
    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]
    # I have to remove the first seq_length labels
    # because for one id the first sequence of seq_length size have as target
    # the last label (the previus ones are discarded).
    # All the next id's sequences will have associated step by step one label as target.
    return data_matrix[seq_length:num_elements, :]


# generate labels
label_array = gen_labels(train_df, sequence_length, ['mosun'])


##label_array = np.concatenate(label_gen).astype(np.float32)
print("训练数据标签的格式")
print(label_array.shape)
#print(label_array)

##################################
# Modeling
##################################
##这里
def r2_keras(y_true, y_pred):
    """Coefficient of Determination
    """
    SS_res =  K.sum(K.square( y_true - y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

# Next, we build a deep network.
# The first layer is an LSTM layer with 100 units followed by another LSTM layer with 50 units.
# Dropout is also applied after each LSTM layer to control overfitting.
# Final layer is a Dense output layer with single unit and linear activation since this is a regression problem.
nb_features = seq_array.shape[2] ##获取第三个维度的大小，也就是特征数
nb_out = label_array.shape[1] ## label只有一个维度

model = Sequential() ## keras的顺序模型
model.add(LSTM(
         input_shape=(sequence_length, nb_features), ##50*25
         units=100, ##长度为100
         return_sequences=True)) ##返回序列
model.add(Dropout(0.2))
model.add(LSTM(
          units=50,
          return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=nb_out)) ##
model.add(Activation("linear")) ##这里.......
model.compile(loss='mean_squared_error', optimizer='rmsprop',metrics=['mae',r2_keras]) ##metrics这一块.............

print(model.summary())


# fit the network
history = model.fit(seq_array, label_array, epochs=30, batch_size=1, validation_split=0.05, verbose=2,
          callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min'),
                       keras.callbacks.ModelCheckpoint(model_path,monitor='val_loss', save_best_only=True, mode='min', verbose=0)]
          )

# list all data in history
print(history.history.keys())

# Save the training history
import pickle

filename = open("./rul_point_regression_model_history", "wb")
pickle.dump(history.history, filename)
filename.close()

history_file = open('./rul_point_regression_model_history', 'rb')
saved_history = pickle.load(history_file)
history_file.close()


# Save model to disk
# serialize model to JSON
model_json = model.to_json()
with open("./rul_point_regression_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("./rul_point_regression_model.h5")
print("Saved model to disk")
'''
'''
###########################################
# Load model and training history from disk
###########################################
import pickle
history_file = open("./rul_point_regression_model_history","rb")
saved_history = pickle.load(history_file)
history_file.close()

# Load model from disk
from keras.models import model_from_json
# load json and create model
json_file = open("./rul_point_regression_model.json", "r")
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("./rul_point_regression_model.h5")
print("Loaded model from disk")

# summarize history for R^2
## R^2 MAE 这些..............................................................
fig_acc = plt.figure(figsize=(10, 5))
plt.plot(saved_history['r2_keras'])
plt.plot(saved_history['val_r2_keras'])
plt.title('model r^2')
plt.ylabel('R^2')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
fig_acc.savefig("./model_r2.png")

# summarize history for MAE
fig_acc = plt.figure(figsize=(10, 10))
plt.plot(saved_history['mean_absolute_error'])
plt.plot(saved_history['val_mean_absolute_error'])
plt.title('model MAE')
plt.ylabel('MAE')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
fig_acc.savefig("./model_mae.png")

# summarize history for Loss
fig_acc = plt.figure(figsize=(10, 5))
plt.plot(saved_history['loss'])
plt.plot(saved_history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
fig_acc.savefig("./model_regression_loss.png")


# training metrics
scores = model.evaluate(seq_array, label_array, verbose=1, batch_size=200)
print('\nMAE: {}'.format(scores[1]))
print('\nR^2: {}'.format(scores[2]))

y_pred = loaded_model.predict(seq_array,verbose=1, batch_size=200)
test_set = pd.DataFrame(y_pred)  ##将预测出来的数据组成dataframe
test_set.to_csv('./result.csv', index = None) ##保存成文件
y_true = label_array

##
predictValue = pd.read_csv('./result.csv', sep=",", header=None)
predictValue.columns=["max"]
fig_test= plt.figure(figsize=(10, 5))
plt.plot(predictValue["max"])
plt.plot(y_true)
plt.show()