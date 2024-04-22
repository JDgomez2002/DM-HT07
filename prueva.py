import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Lectura de datos
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Reemplaza NaN con 0 en la columna 'tipoDeCasa'
train['tipoDeCasa'] = train['tipoDeCasa'].fillna(0).astype(int)

# Eliminación de valores faltantes y conversión de tipos de datos
train.fillna(0, inplace=True)
train['tipoDeCasa'] = pd.cut(train['SalePrice'], bins=[0, 145000, 205000, 410000], labels=[1, 2, 3])
train = train.astype({'tipoDeCasa': 'int'})
train = train.select_dtypes(exclude=['object'])

# División de datos en conjuntos de entrenamiento y prueba
X = train.drop(columns=['tipoDeCasa'])
y = train['tipoDeCasa']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Estandarizar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelos de SVM
modeloSVM_L1 = svm.SVC(kernel='linear', C=32)
modeloSVM_L1.fit(X_train_scaled, y_train)

modeloSVM_R1 = svm.SVC(kernel='rbf', gamma=0.005, C=1)
modeloSVM_R1.fit(X_train_scaled, y_train)

modeloSVM_P1 = svm.SVC(kernel='poly', gamma=1, coef0=1, degree=8, C=1)
modeloSVM_P1.fit(X_train_scaled, y_train)

# Predicciones de SVM
prediccionL1 = modeloSVM_L1.predict(X_test_scaled)
prediccionR1 = modeloSVM_R1.predict(X_test_scaled)
prediccionP1 = modeloSVM_P1.predict(X_test_scaled)

# Matrices de confusión
cmL1 = confusion_matrix(y_test, prediccionL1)
cmR1 = confusion_matrix(y_test, prediccionR1)
cmP1 = confusion_matrix(y_test, prediccionP1)

# Modelos de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

# Evaluación del modelo de regresión lineal
mse = np.mean((y_test - pred) ** 2)
