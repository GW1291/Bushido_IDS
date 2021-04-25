import pre_processing
import keras
tdata = pre_processing.Data()

model = keras.Sequential()

model.add(keras.layers.Dense(64,input_shape=(114,) ,activation="relu", name="dense_0"))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Dense(64,activation="relu", name="dense_1"))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Dense(37,activation="softmax", name="predict"))
model.summary()

#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(
    tdata.ProcessedData._X_train_dataframe.values,
    tdata.ProcessedData._y_train_dataframe.values,
    epochs=3,batch_size=64,
    validation_data=(tdata.ProcessedData._X_test_dataframe.values,tdata.ProcessedData._y_test_dataframe.values)
    )

#rint("Evaluate")
#result = model.evaluate(tdata.ProcessedData._testing_dataset)