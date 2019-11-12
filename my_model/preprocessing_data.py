import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from PIL import Image
import numpy as np
import keras
import random
from sklearn.model_selection import train_test_split


def preprocessing_data():
    print("============================================Start Handle Data============================================")
    size = (128, 128)
    cat_img_list = []
    dog_img_list = []

    base_path = r'/Users/shawnwu4mac/PycharmProjects/DL/kaggle_catsanddogs/PetImages'

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".jpg"):
                filename = os.path.join(root, file)
                file_size = os.path.getsize(filename)
                class_name = os.path.basename(root)

                if file_size >= 2048:
                    im = Image.open(filename)
                    if im.mode == 'RGB':
                        im = im.resize(size, Image.BILINEAR)
                        imarray = np.array(im)

                        if class_name == 'Cat':
                            cat_img_list.append(imarray)
                        elif class_name == "Dog":
                            dog_img_list.append(imarray)

    cat_img_arr = np.asarray(cat_img_list)
    dog_img_arr = np.asarray(dog_img_list)

    cat_img_label = np.ones(cat_img_arr.shape[0])*0
    dog_img_label = np.ones(dog_img_arr.shape[0])*1

    img_arr = np.concatenate((cat_img_arr, dog_img_arr))
    img_label = np.concatenate((cat_img_label, dog_img_label))

    img_label = keras.utils.to_categorical(img_label, num_classes=2)

    temp = list(zip(img_arr, img_label))
    random.shuffle(temp)

    img_arr, img_label = zip(*temp)

    img_arr = np.asarray(img_arr)
    img_label = np.asarray(img_label)

    # print(img_arr)

    # print(img_label)

    train_data, test_data, train_label, test_label = train_test_split(img_arr, img_label, test_size=0.2, random_state=42)


    print("============================================End Hadle Data============================================")


    return train_data, test_data, train_label, test_label
