from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def save_data_csv(data, labels):
    data['label'] = labels
    data_train, data_test = data[:42000], data[42000:]
    data_train.to_csv('data/train.csv', index=False)
    data_test.iloc[:, :-1].to_csv('data/test.csv', index=False)
    data_test.loc[:, 'label'].to_csv('data/test_labels.csv', index=False)


mnist = fetch_openml('mnist_784', version=1)
X, y = mnist['data'], mnist['target']

# Save data into CSV
save_data_csv(X, y)


some_digit = X.iloc[0]
some_digit_image = some_digit.values.reshape(28, 28)
plt.imshow(some_digit_image, cmap = mpl.cm.binary, interpolation='nearest')
plt.axis('off')
plt.show()

print(y[0])  # 5

y = y.astype(np.uint8)

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

#  Multiclass Classification

# SGDC Classifier
sgd_clf = SGDClassifier(random_state=42, n_jobs=-1)
sgd_clf.fit(X_train, y_train)
sgd_clf.predict([some_digit])  # 3

# One vs One version
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42, n_jobs=-1))
ovo_clf.fit(X_train, y_train)
ovo_clf.predict([some_digit])  # 5

# Random Forest Classifier
forest_clf = RandomForestClassifier(random_state=42, n_jobs=-1)
forest_clf.fit(X_train, y_train)
forest_clf.predict([some_digit])  # 5
forest_clf.predict_proba([some_digit])

# Evaluate Models
cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1)  # [0.87365, 0.85835, 0.8689]
cross_val_score(forest_clf, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1)  # [0.9646 , 0.96255, 0.9666]

# Evaluate after scaling inputs
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring='accuracy', n_jobs=-1)  # [0.8983, 0.891 , 0.9018]
cross_val_score(forest_clf, X_train_scaled, y_train, cv=3, scoring='accuracy', n_jobs=-1)  # [0.96445, 0.96255, 0.96645]
