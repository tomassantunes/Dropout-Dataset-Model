import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

class modelo:

    # Remover a coluna 'id' (não é relevante)
    def pre_processing_data(data_train):
        X = data_train.drop('Failure', axis=1).drop('Id', axis=1)
        y = data_train.Failure

        return X, y

    # Os 2 atributos escolhidos serão a média dos créditos
    # feitos por semestre e a média dos créditos feitos
    def pre_processing_data_average(data_train):
        classifications = data_train.drop('Failure', axis=1).drop('Id', axis=1).drop('Program', axis=1).drop(
            list(data_train.filter(regex='enrol')), axis=1).drop(list(data_train.filter(regex='complete')), axis=1)

        ects = data_train.drop('Failure', axis=1).drop('Id', axis=1).drop('Program', axis=1).drop(list(
            data_train.filter(regex='enrol')), axis=1).drop(list(data_train.filter(regex='grade')), axis=1)

        data_train['Classifications_mean'] = classifications.mean(axis=1)

        data_train['ects'] = ects.mean(axis=1)

        X = pd.concat([data_train.pop(x) for x in ['Classifications_mean', 'ects']], axis=1)
        y = data_train.Failure

        return X, y

    def predict(X_train, X_test, y_train, y_test):
        # definir os possíveis parâmetros
        param_grid = {'max_depth': [1, 2, 4, 6, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                      'min_samples_split': [2, 4, 6, 8, 10, 12, 13, 14, 15]}

        grid = GridSearchCV(DecisionTreeClassifier(), param_grid,
                            refit=True, verbose=0, n_jobs=-1, scoring='recall')

        # fit do modelo para procura com GridSearch
        grid.fit(X_train, y_train)

        # mostrar o melhor parámetro depois do tuning
        print(grid.best_params_)
        grid_predictions = grid.predict(X_test)

        print(
            f"Precisão: {precision_score(y_test, grid_predictions, average=None)[1]}")

        print(
            f"Cobertura: {recall_score(y_test, grid_predictions, average=None)[1]}")

        # mostrar tabela de classificações
        print(classification_report(y_test, grid_predictions, zero_division=1))

data_train = pd.read_csv("test-files/dropout-trabalho2.csv")
modelo = modelo

# Primeira Parte
print("Parte 1:")
X, y= modelo.pre_processing_data(data_train)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, random_state = 1) 
modelo.predict(X_train, X_test, y_train, y_test)

# Segunda parte
print("Parte 2:")
X, y = modelo.pre_processing_data_average(data_train)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, random_state = 1) 
modelo.predict(X_train, X_test, y_train, y_test)
