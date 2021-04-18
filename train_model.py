from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
import pickle


mnist = fetch_openml('mnist_784', version=1)
X, y = mnist['data'], mnist['target']

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

knn = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
knn.fit(X_train, y_train)

pickle.dump(knn, open("knn_model.sav", 'wb'))
