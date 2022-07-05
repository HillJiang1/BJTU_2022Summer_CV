# -*- coding: utf-8 -*-

'''
使用CNN做表情识别
'''

# 导入包
from oldcare.preprocessing import SimplePreprocessor
from oldcare.datasets import SimpleDatasetLoader
from imutils import paths
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
#from keras.optimizers import SGD
from keras.optimizers import *
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

from lenet import LeNet

# 全局变量
dataset_path = 'images/mood'
accuracy_plot_path = 'plots/accuracy_cnn.png'
loss_plot_path = 'plots/loss_cnn.png'
output_model_path = 'models/face_expression_cnn.hdf5'

# 全局常量
TARGET_IMAGE_WIDTH = 28
TARGET_IMAGE_HEIGHT = 28
LR = 0.001 # 学习率
BATCH_SIZE = 128
EPOCHS = 40

################################################
# 第一部分：数据预处理

# initialize the image preprocessor and datasetloader
sp = SimplePreprocessor(TARGET_IMAGE_WIDTH, TARGET_IMAGE_HEIGHT)
sdl = SimpleDatasetLoader(preprocessors=[sp])

# Load images
print("[INFO] 导入图像...")
image_paths = list(paths.list_images(dataset_path)) # path included
(X, y) = sdl.load(image_paths, verbose=500, grayscale = True)

# Flatten (reshape the data matrix)
# convert from (13164,TARGET_IMAGE_WIDTH,TARGET_IMAGE_HEIGHT) 
#into (13164,TARGET_IMAGE_WIDTH*TARGET_IMAGE_HEIGHT)
X = X.reshape((X.shape[0], TARGET_IMAGE_WIDTH*TARGET_IMAGE_HEIGHT)) 
X = X.astype("float") / 255.0 # 特征缩放，是非常重要的步骤


X = X.reshape((X.shape[0], 28, 28, 1))


# Show some information on memory consumption of the images
print("[INFO] features matrix: {:.1f}MB"
      .format(X.nbytes / (1024 * 1024.0)))

# Label encoder
le = LabelEncoder()
y = to_categorical(le.fit_transform(y), 2)
print(le.classes_)

# 拆分数据集
(X_train, X_test, y_train, y_test) = train_test_split(X, y, 
                                                      test_size=0.25, 
                                                      random_state=42)

################################################3
# 第二部分：创建并训练模型

print('[INFO] 训练模型...')
opt = Adam(lr = LR)
model = LeNet.build(28,28,2,'',1)
model.compile(loss = 'categorical_crossentropy', 
              optimizer=opt, metrics = ['accuracy'])

H = model.fit(X_train, y_train, validation_data=(X_test, y_test),
	epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

################################################
# 第三部分：评估模型

# 画出accuracy曲线
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(1, EPOCHS+1), H.history["acc"], label="train_acc")
plt.plot(np.arange(1, EPOCHS+1), H.history["val_acc"],label="val_acc")
plt.title("Training Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(accuracy_plot_path)

# 画出loss曲线
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(1, EPOCHS+1),H.history["loss"],label="train_loss")
plt.plot(np.arange(1,EPOCHS+1),H.history["val_loss"],label="val_loss")
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend()
plt.savefig(loss_plot_path)

# 打印分类报告
label_names = le.classes_.tolist()
print("[INFO] 评估模型...")
predictions = model.predict(X_test, batch_size=BATCH_SIZE)
print(classification_report(y_test.argmax(axis=1),
	predictions.argmax(axis=1), target_names=label_names))


################################################
# 第四部分：保存模型
model.save(output_model_path)

