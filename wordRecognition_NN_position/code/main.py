import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold,StratifiedKFold
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout,Conv2D,MaxPooling2D,Flatten,LSTM,SimpleRNN,Embedding
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.metrics import binary_accuracy
import keras
import numpy as np
from sklearn.model_selection import train_test_split
import datetime
import tensorflow as tf
import math





def training_vis(hist):
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    acc = hist.history['acc']
    val_acc = hist.history['val_acc']

    # make a figure
    fig = plt.figure(figsize=(8,4))
    # plot loss
    ax1 = fig.add_subplot(121)
    ax1.plot(loss,label='train_loss')
    ax1.plot(val_loss,label='val_loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss on Training and Validation Data')
    ax1.legend()
    # plot acc
    ax2 = fig.add_subplot(122)
    ax2.plot(acc,label='train_acc')
    ax2.plot(val_acc,label='val_acc')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy  on Training and Validation Data')
    ax2.legend()
    plt.tight_layout()
    plt.show()

###neural network model
def nn_model(train_x,train_y,test_x):
	model = Sequential()


	model.add(Dense(500,input_shape = (train_x.shape[-1],),init='uniform',activation = 'relu'))
	model.add(Dropout(0.5))
	model.add(Dense(200,init='uniform',activation = 'relu'))
	model.add(Dropout(0.5))
	model.add(Dense(100, init='uniform', activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(30,init='uniform',activation='softmax'))


	model.summary()
	#earlystop = EarlyStopping(monitor = 'loss',patience = 3)
	model.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])

	hist = model.fit(train_x,train_y,
		# validation_data = (val_x,val_y),
		validation_split = 0.1,
		batch_size = 20,
		#callbacks = [earlystop],
		epochs = 80)
	# visualize the procedure
	#training_vis(hist)
	y_pre = model.predict(test_x)
	#y_pre = np.argmax(y_pre,1)
	# print(y_pre[:3])
	training_vis(hist)
	return y_pre


def main(flag):

	if flag == 1:#训练数据集较小的train的开始位置label即trainpre.csv

		train_dir = "../data/trainpre.csv"
		test_dir = "../data/test.csv"
		df_train = pd.read_csv(train_dir)
		df_test = pd.read_csv(test_dir)
		sub = pd.DataFrame()
		sub['id'] = df_test['id']
		df = pd.concat([df_train.iloc[:, 1:-1], df_test.iloc[:, 1:]])
		y = df_train['label']
		trian_row = df_train.shape[0]
		print('columns ', df.columns)
		print("df head")
		print(df.head())
		gene_map = {"a": 0, "à": 0, "b": 1, "c": 2, "d": 3, "e": 4, "é": 4, "è": 4, "f": 5, "g": 6, "h": 7, "i": 8,
					"j": 9, "k": 10, "l": 11, "m": 12, "n": 13, "o": 14, "p": 15, "q": 16, "r": 17, "s": 18, "t": 19,
					"u": 20, "v": 21, "w": 22, "x": 23, "y": 24, "z": 25}
		df['sequence'] = df['sequence'].map(list)
		df['sequence'] = df['sequence'].map(lambda x: [gene_map[i] for i in x])
		for i in range(30):
			df['sequence' + str(i)] = list(map(lambda x: x[i], df['sequence']))
		del df['sequence']
		print("after pre ")
		print(df.head())
		train = df.iloc[:trian_row:, :]
		test = df.iloc[trian_row:, :]
		y = to_categorical(y, 30)
		print('train shape is', train.shape)
		print('test shape is', test.shape)
		predict_label = nn_model(train, y, test)
		prediction = []
		for i in range(predict_label.shape[0]):
			prediction.append(np.argmax(predict_label[i][:]) + 1)
		print(prediction[0:3])
		sub['label'] = prediction

		# sub_name = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
		# sub.to_csv("../result/" + sub_name +".csv",index = None)

		# 用train.csv训练出来的预测结果
		sub.to_csv("../result/testpre(train).csv", index=None)
		# 用bigTrain.csv训练出来的预测结果
		# sub.to_csv("../result/resultpredicttest(big_test).csv", index=None)
		print(sub.head())

	elif flag==2:#训练train数据的结束标签即trainlast.csv文件
		train_dir = "../data/trainlast.csv"
		test_dir = "../data/test.csv"
		df_train = pd.read_csv(train_dir)
		df_test = pd.read_csv(test_dir)
		sub = pd.DataFrame()
		sub['id'] = df_test['id']
		df = pd.concat([df_train.iloc[:, 1:-1], df_test.iloc[:, 1:]])
		y = df_train['label']
		trian_row = df_train.shape[0]
		print('columns ', df.columns)
		print("df head")
		print(df.head())
		gene_map = {"a": 0, "à": 0, "b": 1, "c": 2, "d": 3, "e": 4, "é": 4, "è": 4, "f": 5, "g": 6, "h": 7, "i": 8,"j": 9, "k": 10, "l": 11, "m": 12, "n": 13, "o": 14, "p": 15, "q": 16, "r": 17, "s": 18, "t": 19,"u": 20, "v": 21, "w": 22, "x": 23, "y": 24, "z": 25}
		df['sequence'] = df['sequence'].map(list)
		df['sequence'] = df['sequence'].map(lambda x: [gene_map[i] for i in x])
		for i in range(30):
			df['sequence' + str(i)] = list(map(lambda x: x[i], df['sequence']))
		del df['sequence']
		print("after pre ")
		print(df.head())
		train = df.iloc[:trian_row:, :]
		test = df.iloc[trian_row:, :]
		y = to_categorical(y, 30)
		print('train shape is', train.shape)
		print('test shape is', test.shape)
		predict_label = nn_model(train, y, test)
		prediction = []
		for i in range(predict_label.shape[0]):
			prediction.append(np.argmax(predict_label[i][:]) + 1)
		print(prediction[0:3])
		sub['label'] = prediction

		# sub_name = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
		# sub.to_csv("../result/" + sub_name +".csv",index = None)

		# 用train.csv训练出来的预测结果
		sub.to_csv("../result/testlast(train).csv", index=None)
		# 用bigTrain.csv训练出来的预测结果
		# sub.to_csv("../result/resultpredicttest(big_test).csv", index=None)
		print(sub.head())

	elif flag==3: #训练bigtrain训练数据的开始标签bigTrainpre.csv
		train_dir = "../data/bigTrainpre.csv"
		test_dir = "../data/test.csv"
		df_train = pd.read_csv(train_dir)
		df_test = pd.read_csv(test_dir)
		sub = pd.DataFrame()
		sub['id'] = df_test['id']
		df = pd.concat([df_train.iloc[:, 1:-1], df_test.iloc[:, 1:]])
		y = df_train['label']
		trian_row = df_train.shape[0]
		print('columns ', df.columns)
		print("df head")
		print(df.head())
		gene_map = {"a": 0, "à": 0, "b": 1, "c": 2, "d": 3, "e": 4, "é": 4, "è": 4, "f": 5, "g": 6, "h": 7, "i": 8,
					"j": 9, "k": 10, "l": 11, "m": 12, "n": 13, "o": 14, "p": 15, "q": 16, "r": 17, "s": 18, "t": 19,
					"u": 20, "v": 21, "w": 22, "x": 23, "y": 24, "z": 25}
		df['sequence'] = df['sequence'].map(list)
		df['sequence'] = df['sequence'].map(lambda x: [gene_map[i] for i in x])
		for i in range(30):
			df['sequence' + str(i)] = list(map(lambda x: x[i], df['sequence']))
		del df['sequence']
		print("after pre ")
		print(df.head())
		train = df.iloc[:trian_row:, :]
		test = df.iloc[trian_row:, :]
		y = to_categorical(y, 30)
		print('train shape is', train.shape)
		print('test shape is', test.shape)
		predict_label = nn_model(train, y, test)
		prediction = []
		for i in range(predict_label.shape[0]):
			prediction.append(np.argmax(predict_label[i][:]) + 1)
		print(prediction[0:3])
		sub['label'] = prediction

		# sub_name = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
		# sub.to_csv("../result/" + sub_name +".csv",index = None)

		# 用train.csv训练出来的预测结果
		sub.to_csv("../result/testpre(bigTrain).csv", index=None)
		# 用bigTrain.csv训练出来的预测结果
		# sub.to_csv("../result/resultpredicttest(big_test).csv", index=None)
		print(sub.head())

	elif flag==4:
		train_dir = "../data/bigTrainlast.csv"
		test_dir = "../data/test.csv"
		df_train = pd.read_csv(train_dir)
		df_test = pd.read_csv(test_dir)
		sub = pd.DataFrame()
		sub['id'] = df_test['id']
		df = pd.concat([df_train.iloc[:, 1:-1], df_test.iloc[:, 1:]])
		y = df_train['label']
		trian_row = df_train.shape[0]
		print('columns ', df.columns)
		print("df head")
		print(df.head())
		gene_map = {"a": 0, "à": 0, "b": 1, "c": 2, "d": 3, "e": 4, "é": 4, "è": 4, "f": 5, "g": 6, "h": 7, "i": 8,
					"j": 9, "k": 10, "l": 11, "m": 12, "n": 13, "o": 14, "p": 15, "q": 16, "r": 17, "s": 18, "t": 19,
					"u": 20, "v": 21, "w": 22, "x": 23, "y": 24, "z": 25}
		df['sequence'] = df['sequence'].map(list)
		df['sequence'] = df['sequence'].map(lambda x: [gene_map[i] for i in x])
		for i in range(30):
			df['sequence' + str(i)] = list(map(lambda x: x[i], df['sequence']))
		del df['sequence']
		print("after pre ")
		print(df.head())
		train = df.iloc[:trian_row:, :]
		test = df.iloc[trian_row:, :]
		y = to_categorical(y, 30)
		print('train shape is', train.shape)
		print('test shape is', test.shape)
		predict_label = nn_model(train, y, test)
		prediction = []
		for i in range(predict_label.shape[0]):
			prediction.append(np.argmax(predict_label[i][:]) + 1)
		print(prediction[0:3])
		sub['label'] = prediction

		# sub_name = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
		# sub.to_csv("../result/" + sub_name +".csv",index = None)

		# 用train.csv训练出来的预测结果
		sub.to_csv("../result/testlast(bigTrain).csv", index=None)
		# 用bigTrain.csv训练出来的预测结果
		# sub.to_csv("../result/resultpredicttest(big_test).csv", index=None)
		print(sub.head())
	else:
		print('wrong option!')


if __name__ == '__main__':
	main(3)
	main(4)
