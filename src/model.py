import pre_processing
import keras
import tensorflow as tf
import numpy as np
import sklearn
import tensorflow_addons as tfa
import tensorflow_datasets as tfd

class Model():
    def __init__(self):
        self._Model = None
        self._Data = None

        ##compostion
        self._Data = pre_processing.Data()

    def set_ANN_model(self) -> 'assign':
        self._Model = keras.Sequential()
        self._Model.add(keras.layers.BatchNormalization())
        self._Model.add(keras.layers.Dense(64,activation="relu", name="dense_0"))
        self._Model.add(keras.layers.Dropout(0.2))

        self._Model.add(keras.layers.Dense(38,activation="softmax", name="predict"))
        self._Model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['CategoricalAccuracy'])
        #self._Model.summary()
    
    def set_small_RNN_model(self) -> 'assign':
        self._Model = keras.Sequential()
        
        self._Model.add(keras.layers.BatchNormalization())
        self._Model.add(keras.layers.LSTM(256,return_sequences=True,activation="tanh"))
        self._Model.add(keras.layers.Dropout(0.2))

        self._Model.add(keras.layers.LSTM(128,activation="tanh"))
        self._Model.add(keras.layers.Dropout(0.2))

        self._Model.add(keras.layers.Dense(64,activation="tanh"))
        self._Model.add(keras.layers.Dropout(0.2))

        self._Model.add(keras.layers.Dense(38,activation="softmax", name="predict"))
        self._Model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['CategoricalAccuracy','Recall','Precision',tf.keras.metrics.AUC(multi_label=True)])

    def set_BIG_RNN_model(self) -> 'assign':
        self._Model = keras.Sequential()
        
        self._Model.add(keras.layers.BatchNormalization())
        self._Model.add(keras.layers.LSTM(512,return_sequences=True,activation="tanh"))
        self._Model.add(keras.layers.Dropout(0.2))

        self._Model.add(keras.layers.LSTM(256,return_sequences=True,activation="tanh"))
        self._Model.add(keras.layers.Dropout(0.2))

        self._Model.add(keras.layers.LSTM(64,activation="tanh"))
        self._Model.add(keras.layers.Dropout(0.2))

        self._Model.add(keras.layers.Dense(64,activation="tanh"))
        self._Model.add(keras.layers.Dropout(0.2))

        self._Model.add(keras.layers.Dense(38,activation="softmax", name="predict"))
        self._Model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['CategoricalAccuracy','Recall','Precision',tf.keras.metrics.AUC(multi_label=True)])

    def fit_model(self):
        self._Model.fit(
            self._Data.ProcessedData._X_train_dataframe.values,
            self._Data.ProcessedData._y_train_dataframe.values,
            epochs=5,batch_size=64,
            validation_data=(self._Data.ProcessedData._X_test_dataframe.values,
                             self._Data.ProcessedData._y_test_dataframe.values)
            )

    def fit_model_tf(self):
        X_train = self._Data.ProcessedData._X_train_dataframe.values
        y_train = self._Data.ProcessedData._y_train_dataframe.values
        tf_train_dataset = keras.preprocessing.timeseries_dataset_from_array(X_train,y_train,10)

        X_test = self._Data.ProcessedData._X_test_dataframe.values
        y_test = self._Data.ProcessedData._y_test_dataframe.values
        tf_test_dataset = keras.preprocessing.timeseries_dataset_from_array(X_test,y_test,10)

        self._Model.fit(
            tf_train_dataset,
            epochs=5,batch_size=64,
            validation_data=tf_test_dataset
            )

    def predict(self,x_new_array):
        return self._Model.predict(x_new_array)








#debug info
"""
tModel = Model()
#x_array = tModel._Data.ProcessedData._X_train_dataframe.values
#y_array = tModel._Data.ProcessedData._y_train_dataframe.values

#tModel.set_ANN_model()
tModel.set_small_RNN_model()
#tModel.set_BIG_RNN_model()
#tModel.fit_model()
tModel.fit_model_tf()

print(tModel._Data.ProcessedData._X_train_dataframe.dtypes)
print(tModel._Data.ProcessedData._y_train_dataframe.dtypes)

x_vector = tModel._Data.ProcessedData._X_train_dataframe.iloc[2]
y_vector = tModel._Data.ProcessedData._y_train_dataframe.iloc[2]


print(y_vector.values.shape)
print(x_vector.to_frame().shape)

print(y_vector)
print(f'true_label: {y_vector.idxmax(0)}')


X_test = tModel._Data.ProcessedData._X_test_dataframe.values
y_test = tModel._Data.ProcessedData._y_test_dataframe.values
#tf_test_dataset = keras.preprocessing.timeseries_dataset_from_array(X_test,y_test,10)

y_predict = tModel.predict(X_test)
print(y_predict.shape)
print(y_test.shape)


metric = tfa.metrics.MultiLabelConfusionMatrix(num_classes=38)
metric.update_state(y_test, y_predict)
result = metric.result()
print(result)

#X_test = tModel._Data.ProcessedData._X_test_dataframe.values
#y_test = tModel._Data.ProcessedData._y_test_dataframe.values
#tf_test_dataset = keras.preprocessing.timeseries_dataset_from_array(X_test,y_test,10)


#tfa.metrics.MultiLabelConfusionMatrix(tf_test_dataset)


for i in range(20):
    true = tModel._Data.ProcessedData._y_test_dataframe.values[i]
    predict = y_predict[i]
    print(f'true:{true}\n')
    print(f'max:{np.amax(true)} at location:{np.where(true == np.amax(true))}')
    print(f'predicted:{y_predict[i]}\n')
    print(f'max:{np.amax(predict)} at location:{np.where(predict == np.amax(predict))}')
    print(f'predicted_rounded:{np.round(y_predict[i])}\n')
"""