import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = 'data/train'
val_dir = 'data/test'
img_size=48
batch_size=64
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size,img_size),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')
val_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(img_size,img_size),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')
for i in os.listdir("data/train/"):
    print(str(len(os.listdir("data/train/"+i))) +" "+ i +" images")

for i in os.listdir("data/test/"):
    print(str(len(os.listdir("data/test/"+i))) +" "+ i +" images")

emotion_model = Sequential()
#Convolutional layer 1
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))#output=(48-3+0)/1+1=46

#Convolutional layer 2
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))#output=(46-3+0)/1+1=44
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))#output=divide input by 2 it means 22
emotion_model.add(Dropout(0.25))#reduce 25% module at a time of output

#Convolutional layer 3
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu',input_shape=(48,48,1)))#(22-3+0)/1+1=20
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))#10
emotion_model.add(Dropout(0.25))

#Convolutional layer 4
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))#(10-3+0)/1+1=8
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))#output=4
emotion_model.add(Dropout(0.25))#nothing change

#Fully Connected layer 1
emotion_model.add(Flatten())#here we get multidimention output and pass as linear to the dense so that 4*4*128=2048
emotion_model.add(Dense(1024, activation='relu'))#hddien of 1024 neurons of input 
emotion_model.add(Dropout(0.35))

#Fully Connected layer 2
emotion_model.add(Dense(7, activation='softmax'))#hddien of 7 neurons of input

emotion_model.summary()

emotion_model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0005, decay=1e-6),metrics=['accuracy'])

emotion_model_info = emotion_model.fit_generator(
        train_generator,
        steps_per_epoch=28709 // batch_size,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=7178 // batch_size)

acc = emotion_model_info.history['accuracy']
val_acc = emotion_model_info.history['val_accuracy']

loss = emotion_model_info.history['loss']
val_loss = emotion_model_info.history['val_loss']

epochs_range = range(10)

plt.figure(figsize=(12, 12))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training')
plt.plot(epochs_range, val_acc, label='Validation')
plt.legend(loc='lower right')
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training')
plt.plot(epochs_range, val_loss, label='Validation')
plt.legend(loc='upper right')
plt.title('Loss')
plt.show()

emotion_model.save_weights('emotion_model.h5')

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Natural", 5: "Sad", 6: "Surprised"}
cur_path = os.path.dirname(os.path.abspath(__file__))

cap = cv2.VideoCapture(0)
while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    if not ret:
        break
    bounding_box = cv2.CascadeClassifier(cur_path+'/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Video', cv2.resize(frame,(600,500),interpolation = cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
