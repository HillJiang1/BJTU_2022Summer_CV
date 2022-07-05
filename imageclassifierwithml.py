# -*- coding: utf-8 -*-

'''
image classification with knn

'''

from oldcare.preprocessing import SimplePreprocessor 
from oldcare.datasets import SimpleDatasetLoader
from imutils import paths
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


################################################3
# 第一部分：数据预处理

# 全局变量
dataset_path = 'images'

# 全局常量
N_NEIGHBOURS = 5
TARGET_IMAGE_WIDTH = 32
TARGET_IMAGE_HEIGHT = 32

# initialize the image preprocessor and datasetloader
sp = SimplePreprocessor(TARGET_IMAGE_WIDTH, TARGET_IMAGE_HEIGHT)
sdl = SimpleDatasetLoader(preprocessors=[sp])

# Load images
print("[INFO] loading images...")
image_paths = list(paths.list_images(dataset_path)) # path included
(X, y) = sdl.load(image_paths, verbose=500, grayscale = True)

# Flatten (reshape the data matrix)
# convert from (13164,32,32) into (13164,32*32)
X = X.reshape((X.shape[0], TARGET_IMAGE_WIDTH*TARGET_IMAGE_HEIGHT)) 

# Show some information on memory consumption of the images
print("[INFO] features matrix: {:.1f}MB"
      .format(X.nbytes / (1024 * 1024.0)))

# Label encoder
le = LabelEncoder()
y = le.fit_transform(y)

# Split dataset
(X_train, X_test, y_train, y_test) = train_test_split(X, y, 
                                                      test_size=0.25, 
                                                      random_state=42)

################################################3
# 第二部分：训练模型

# Train model
print("[INFO] evaluating k-NN classifier...")
model = KNeighborsClassifier(n_neighbors= N_NEIGHBOURS, 
                             metric = 'minkowski', p = 2)
model.fit(X_train, y_train)


################################################
# 第三部分：评估模型

# Evaluate model
y_pred = model.predict(X_test)
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
# Report
print(classification_report(y_test, y_pred, target_names=le.classes_))
