from keras.models import Sequential
import numpy as np
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from keras.preprocessing import image

from keras.preprocessing.image import array_to_img
from ann_visualizer.visualize import ann_viz

def create_model(train_data, test_data, train_labels, test_labels):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3), padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add( MaxPooling2D( pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(128, activation='relu'))

    model.add(Dense(2, activation='sigmoid'))

    model.summary()

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_data, train_labels, batch_size=32, epochs=9, shuffle=True, validation_data=(test_data, test_labels))

    model.save('CatDogModel.h5')

    return model


def predict(image_path):
    model = load_model('CatDogModel.h5')

    usable_input_data = handle_image(image_path)

    predictions = model.predict(usable_input_data)

    if predictions[0, 1] >= 0.5:
        print('I am {:.2%} sure this is a Dog'.format(predictions[0, 1]))
    else:
        print('I am {:.2%} sure this is a Cat'.format(1 - predictions[0, 1]))

    plt.imshow( array_to_img( image.load_img(image_path)))
    plt.show()


def handle_image(image_path):
    img = image.load_img( image_path, target_size=(128, 128) )
    array = image.img_to_array( img )
    usable_input_data = np.expand_dims(array, axis=0)

    return usable_input_data
