import pre_processing
import keras
import numpy as np

class Model():
    def __init__(self):
        self._Model = None
        self._Data = None

        ##compostion
        self._Data = pre_processing.Data()

    def set_ANN_model(self) -> 'assign':
        self._Model = keras.Sequential()
        self._Model.add(keras.layers.Dense(64,activation="relu", name="dense_0"))
        self._Model.add(keras.layers.Dropout(0.2))

        self._Model.add(keras.layers.Dense(37,activation="softmax", name="predict"))
        self._Model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy','CategoricalAccuracy','AUC','TruePositives','FalseNegatives','FalsePositives'])
        #self._Model.summary()
    
    def set_small_RNN_model(self) -> 'assign':
        self._Model = keras.Sequential()
        
        self._Model.add(keras.layers.LSTM(128,return_sequences=True))
        self._Model.add(keras.layers.Dropout(0.2))

        self._Model.add(keras.layers.LSTM(64))
        self._Model.add(keras.layers.Dropout(0.2))

        self._Model.add(keras.layers.Dense(64))
        self._Model.add(keras.layers.Dropout(0.2))

        self._Model.add(keras.layers.Dense(37,activation="softmax", name="predict"))
        self._Model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy','CategoricalAccuracy','AUC','TruePositives','FalseNegatives','FalsePositives'])

    def set_BIG_RNN_model(self) -> 'assign':
        self._Model = keras.Sequential()
        
        self._Model.add(keras.layers.LSTM(256,return_sequences=True))
        self._Model.add(keras.layers.Dropout(0.2))

        self._Model.add(keras.layers.LSTM(128,return_sequences=True))
        self._Model.add(keras.layers.Dropout(0.2))

        self._Model.add(keras.layers.LSTM(64))
        self._Model.add(keras.layers.Dropout(0.2))

        self._Model.add(keras.layers.Dense(64))
        self._Model.add(keras.layers.Dropout(0.2))

        self._Model.add(keras.layers.Dense(37,activation="softmax", name="predict"))
        self._Model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy','CategoricalAccuracy','AUC','TruePositives','FalseNegatives','FalsePositives'])

    def fit_model(self):
        self._Model.fit(
            self._Data.ProcessedData._X_train_dataframe.values,
            self._Data.ProcessedData._y_train_dataframe.values,
            epochs=1,batch_size=32,
            validation_data=(self._Data.ProcessedData._X_test_dataframe.values,
                             self._Data.ProcessedData._y_test_dataframe.values)
            )

    def fit_model_tf(self):
        X_train = tModel._Data.ProcessedData._X_train_dataframe.values
        y_train = tModel._Data.ProcessedData._y_train_dataframe.values
        tf_train_dataset = keras.preprocessing.timeseries_dataset_from_array(X_train,y_train,10)

        X_test = tModel._Data.ProcessedData._X_test_dataframe.values
        y_test = tModel._Data.ProcessedData._y_test_dataframe.values
        tf_test_dataset = keras.preprocessing.timeseries_dataset_from_array(X_test,y_test,10)

        self._Model.fit(
            tf_train_dataset,
            epochs=5,batch_size=64,
            validation_data=tf_test_dataset
            )

    def predict(self,x_new_array):
        return self._Model.predict(x_new_array)


tModel = Model()
x_array = tModel._Data.ProcessedData._X_train_dataframe.values
y_array = tModel._Data.ProcessedData._y_train_dataframe.values

print(x_array.shape)
print(y_array.shape)

tModel.set_ANN_model()
#tModel.set_small_RNN_model()
tModel.fit_model()
#tModel.fit_model_tf()

x_vector = tModel._Data.ProcessedData._X_train_dataframe.iloc[:1]
y_vector = tModel._Data.ProcessedData._y_train_dataframe.iloc[1]
print(y_vector)
print(x_vector.columns)
print(x_vector.values.shape)
print(f'true_label: {y_vector.idxmax(1)}')
y_predict_vector = tModel.predict(x_vector.values)
print(y_predict_vector)
print(f'predict_label: {y_predict_vector.idxmax(1)}')