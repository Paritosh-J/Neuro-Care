import os
# import pickle
import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import load_img 
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


for dirname, _, filenames in os.walk('C:/Coding Stuff/Python/python prgs/hackathons/Tumour_detect-main/brain_tumor_dataset'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


encoder = OneHotEncoder()
encoder.fit([[0], [1]])

dt = []
paths = []
final = []

for r, d, f in os.walk(r'C:/Coding Stuff/Python/python prgs/hackathons/Tumour_detect-main/brain_tumor_dataset/yes'):
    for file in f:
        if '.jpg' in file:
            paths.append(os.path.join(r, file))

for path in paths:
    img = Image.open(path)
    img = img.resize((128,128))
    img = np.array(img)
    if(img.shape == (128,128,3)):
        dt.append(np.array(img))
        final.append(encoder.transform([[0]]).toarray())
        
paths = []
for r, d, f in os.walk(r"C:/Coding Stuff/Python/python prgs/hackathons/Tumour_detect-main/brain_tumor_dataset/no"):
    for file in f:
        if '.jpg' in file:
            paths.append(os.path.join(r, file))

for path in paths:
    img = Image.open(path)
    img = img.resize((128,128))
    img = np.array(img)
    if(img.shape == (128,128,3)):
        dt.append(np.array(img))
        final.append(encoder.transform([[1]]).toarray())
        
dt = np.array(dt)
final = np.array(final)
final = final.reshape(139,2)

x_train,x_test,y_train,y_test = train_test_split(dt, final, test_size=0.2, shuffle=True, random_state=0)

# x_train = np.asarray(x_train)
# y_train = np.asarray(y_train)
# x_test = np.asarray(x_test)
# y_test = np.asarray(y_test)

model = Sequential()

model.add(Conv2D(32, kernel_size=(2, 2), input_shape=(128, 128, 3), padding = 'Same'))
model.add(Conv2D(32, kernel_size=(2, 2),  activation ='relu', padding = 'Same'))


model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size = (2,2), activation ='relu', padding = 'Same'))
model.add(Conv2D(64, kernel_size = (2,2), activation ='relu', padding = 'Same'))

model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss = "categorical_crossentropy", optimizer='Adamax')

y_train.shape

history = model.fit(x_train, y_train, epochs = 30, batch_size = 40, verbose = 1,validation_data = (x_test, y_test))

def tumour_detect(new_image):
    img = load_img(new_image, target_size = (224,224))
    x = np.array(img.resize((128,128)))
    x = x.reshape(1,128,128,3)
    res = model.predict_on_batch(x)
    
tumour_detect('C:/Coding Stuff/Python/python prgs/hackathons/Tumour_detect-main/brain_tumor_dataset/yes/Y3.jpg') 

# pickle.dump(model, open('model.pkl','wb'))
# # Loading model to compare the results
# model_f = pickle.load(open('model.pkl','rb'))

model.save('C:/Coding Stuff/Python/python prgs/hackathons/Tumour_detect-main/model.hdf5')