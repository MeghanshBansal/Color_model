import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
import pandas as pd 

gen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)

train = gen.flow_from_directory('color_dataset', subset='training', target_size=(150, 150))
valid = gen.flow_from_directory('color_dataset', subset='validation', target_size=(150, 150))

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=(150, 150, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['Accuracy'])

print(model.summary())

history = model.fit(train, validation_data=valid, epochs=10)

model.save('color.h5')

pd.DataFrame(history.history).plot()