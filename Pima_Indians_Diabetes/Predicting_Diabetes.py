# Neural Network Application: Predicting Diabetes
# HGDIV

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

dataset = pd.read_csv('pima-indians-diabetes.csv')

# preg = Number of times pregnant
# plas = Plasma glucose concentration a 2 hours in an oral glucose tolerance test
# pres = Diastolic blood pressure (mm Hg)
# skin = Triceps skin fold thickness (mm)
# test = 2-Hour serum insulin (mu U/ml)
# mass = Body mass index (weight in kg/(height in m)^2)
# pedi = Diabetes pedigree function
# age = Age (years)
# x is the dataset that contains the independent (predictive) variables
# y is the dataset that contains the outcome variable class

x = dataset.drop(['class'], axis=1)
y = dataset['class']

model = Sequential()
model.add(Dense(8, input_dim=8, activation='relu'))  # Input layer, 8 neurons
model.add(Dense(6, activation='relu'))
model.add(Dense(4, activation='relu'))  # Hidden layer, 4 neurons
model.add(Dense(1, activation='sigmoid'))  # Output layer, 1 neuron
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x, y, epochs=150, verbose=0)  # An epoch is an iteration over the entire x and y data

loss, accuracy = model.evaluate(x, y)
print(accuracy)

yhat = model.predict_classes(x)

dataset['yhat'] = yhat

pd.DataFrame(dataset).to_csv("Diabetes-with-predictions-from-NeuralNet.csv")
