import pandas as pd
from sklearn.model_selection import train_test_split

from trabalho2 import modelo

data_train = pd.read_csv("test-files/dropout-trabalho2.csv")
modelo = modelo

# Primeira Parte
print("Parte 1:")
X, y= modelo.pre_processing_data(data_train)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, random_state = 1) 
modelo.predict(X_train, X_test, y_train, y_test)

# Segunda parte
print("\nParte 2:")
X, y = modelo.pre_processing_data_average(data_train)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, random_state = 1) 
modelo.predict(X_train, X_test, y_train, y_test)

