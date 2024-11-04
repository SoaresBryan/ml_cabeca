import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle
import os

features_list = []
labels_list = []
modelo = None
modelo_treinado = False

def coletar_dados(features, label):
    features_list.append(features)
    labels_list.append(label)
    print(f"Dados coletados para o movimento {label}")

def treinar_modelo():
    global modelo, modelo_treinado

    # Verifica se há dados suficientes para treinamento
    if len(features_list) == 0:
        print("Nenhum dado coletado para treinamento.")
        return

    X = np.array(features_list)
    y = np.array(labels_list)

    # Divisão dos dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    modelo = SVC(kernel='linear')
    modelo.fit(X_train, y_train)
    modelo_treinado = True

    # Cria o diretório models/ se não existir
    os.makedirs('models', exist_ok=True)
    with open('models/modelo_movimentos.pkl', 'wb') as f:
        pickle.dump(modelo, f)
    
    print("Modelo treinado e salvo com sucesso.")
