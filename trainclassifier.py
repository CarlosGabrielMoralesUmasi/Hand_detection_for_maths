import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score

# Cargar el dataset
with open("./extracted_landmarks.pickle", "rb") as f:
    dataset = pickle.load(f)

# Filtrar elementos que no tienen la longitud esperada
dataset["dataset"] = [i for i in dataset["dataset"] if len(i) == 42]
dataset["labels"] = [label for i, label in zip(dataset["dataset"], dataset["labels"]) if len(i) == 42]

# Verificar que el conjunto de datos no esté vacío después de filtrar
if len(dataset["dataset"]) == 0:
    raise ValueError("El conjunto de datos está vacío después de filtrar. Verifica el contenido de 'extracted_landmarks.pickle'.")

# Convertir los datos y etiquetas a arrays de NumPy
data = np.asarray(dataset["dataset"])
labels = np.asarray(dataset["labels"])

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42)

# Entrenar el modelo
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluar el modelo
y_pred = model.predict(X_test)
score = accuracy_score(y_pred, y_test)
print(f'{score * 100}% of samples were classified correctly !')

# Guardar el modelo entrenado
with open("./rf_model.p", "wb") as f:
    pickle.dump({"model": model}, f)
