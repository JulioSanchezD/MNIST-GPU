import cudf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier as K2
from cuml.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import time
import pickle


# Load Data
train = cudf.read_csv('data/train.csv')
train.head()

# Visualize data
samples = train.iloc[5000:5030, 1:].to_pandas().values
plt.figure(figsize=(15, 4.5))
for i in range(30):
    plt.subplot(3, 10, i+1)
    plt.imshow(samples[i].reshape((28, 28)), cmap=plt.cm.binary)
    plt.axis('off')
plt.subplots_adjust(wspace=-0.1, hspace=-0.1)
plt.show()

# Create 20% Validation set
X_train, X_test, y_train, y_test = train_test_split(train.iloc[:, :-1], train.loc[:, 'label'], test_size=0.2, random_state=42)

# Grid Search kNN for optimal k
accs = []
for k in range(3, 22):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_hat = knn.predict(X_test)
    acc = (y_hat.to_array() == y_test.to_array()).sum()/y_test.shape
    print(k, acc)
    accs.append(acc)

# Free memory
del X_train, X_test, y_train, y_test

# Plot grid search results
plt.figure(figsize=(15, 5))
plt.plot(range(3, 22), accs)
plt.title('MNIST kNN k value versus validation acc')
plt.show()

# KFold Grid Search (cross validation
for k in range(3, 6):
    print('k =', k)
    oof = np.zeros(len(train))
    skf = KFold(n_splits=5, shuffle=True, random_state=42)
    for i, (idxT, idxV) in enumerate(skf.split(train.iloc[:, :-1], train.loc[:, 'label'])):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(train.iloc[idxT, :-1], train.loc[idxT, 'label'])
        y_hat = knn.predict(train.iloc[idxV, :-1])
        oof[idxV] = y_hat.to_array()
        acc = (oof[idxV] == train.loc[idxV, 'label'].to_array()).sum() / len(idxV)
        print(' fold =', i, 'acc =', acc)
    acc = (oof == train.loc[:, 'label'].to_array()).sum() / len(train)
    print(' OOF with k =', k, 'ACC =', acc)

# Load test set
test = cudf.read_csv('data/test.csv')
test.head()
test.shape

# Fit kNN Model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train.iloc[:, :-1], train.loc[:, 'label'])

# Predict test data
start_time = time.time()
y_hat = knn.predict(test).to_array()
print("time elapsed: {:.2f}s".format(time.time() - start_time))  # 1.75 seconds

# Save predictions to csv
sub = pd.read_csv('data/test_labels.csv')
sub['prediction'] = y_hat
sub.to_csv('data/test_predictions.csv', index=False)

# Plot predictions histogram
plt.hist(sub['prediction'])
plt.title('Distribution of test predictions')
plt.show()

# Finally, print accuracy
acc = (sub['label'] == sub['prediction']).sum() / sub.shape[0]
print(f"Accuracy: {acc*100:.2f}%")  # 96.73%

# Compare it to CPU execution time
knn = K2(n_neighbors=3, n_jobs=-1)
knn.fit(train.iloc[:, :-1].to_pandas(), train.loc[:, 'label'].to_pandas())

start_time = time.time()
y_hat = knn.predict(test.to_pandas())
print("time elapsed: {:.2f}s".format(time.time() - start_time))  # 23.6 seconds

# Save model
pickle.dump(knn, open("knn_model.sav", 'wb'))