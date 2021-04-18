from PIL import Image
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


# Load Image as PIL Object
img = Image.open("number.png")

# Resize image to mnist data size
img = img.resize((28, 28))

# Convert image to numpy array
img = np.array(img)[:, :, -1]

# Plot image
plt.imshow(img, cmap = mpl.cm.binary, interpolation='nearest')
plt.axis('off')
plt.show()

# Convert image into single array
img = img.reshape(784)

# Predict image using trained KNN model
knn = pickle.load(open("knn_model.sav", 'rb'))
knn.predict([img])