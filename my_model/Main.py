from my_model.model import create_model, predict
from my_model.preprocessing_data import preprocessing_data



# training
train_data, test_data, train_label, test_label = preprocessing_data()

this_model = create_model(train_data, test_data, train_label, test_label)


# predict
predict( '/Users/shawnwu4mac/PycharmProjects/DL/Real_data/fat-cat.jpg')
