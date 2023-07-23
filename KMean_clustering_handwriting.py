import codecademylib3_seaborn
import numpy as np
from matplotlib import pyplot as plt

# importing data
from sklearn import datasets
from sklearn.cluster import KMeans

digits = datasets.load_digits()
# print(digits.target)

# Figure size (width, height)

fig = plt.figure(figsize=(6, 6))

# Adjust the subplots 

fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# For each of the 64 images

for i in range(64):
    # Initialize the subplots: add a subplot in the grid of 8 by 8, at the i+1-th position

    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])

    # Display an image at the i-th position

    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')

    # Label the image with the target value

    ax.text(0, 7, str(digits.target[i]))

plt.show()

# building a model
model = KMeans(n_clusters=10, random_state=42)

# fitting the data
model.fit(digits.data)

# adding new figure
fig = plt.figure(figsize=(8, 3))

fig.suptitle('Cluser Center Images', fontsize=14, fontweight='bold')

for i in range(10):
    # initialize subplots
    ax = fig.add_subplot(2, 5, 1 + i)

    # display images
    ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)

plt.show()

new_samples = np.array([
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.60, 7.33, 0.99, 0.00, 0.00, 0.00, 0.00, 0.00, 3.05, 7.63,
     1.53, 0.00, 1.15, 0.23, 0.00, 0.00, 5.95, 7.33, 1.60, 2.98, 7.55, 3.59, 0.00, 0.00, 5.11, 7.63, 7.62, 7.62, 7.63,
     2.13, 0.00, 0.00, 0.00, 0.92, 1.52, 3.35, 7.63, 1.52, 0.00, 0.00, 0.00, 0.00, 0.00, 1.67, 7.63, 2.60, 0.00, 0.00,
     0.00, 0.00, 0.00, 0.45, 7.17, 2.28, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.08, 4.43, 6.41, 0.38, 0.00, 0.00, 0.00, 0.00,
     4.50, 7.63, 4.27, 0.08, 0.00, 0.00, 0.00, 1.30, 7.56, 4.50, 0.00, 0.00, 0.00, 0.00, 0.00, 2.28, 7.63, 1.68, 0.00,
     0.00, 0.00, 0.00, 0.00, 2.28, 7.63, 6.87, 6.88, 3.98, 0.00, 0.00, 0.00, 1.37, 7.63, 7.32, 7.17, 6.10, 0.00, 0.00,
     0.00, 0.00, 4.27, 7.62, 7.62, 3.81, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.15, 6.03, 6.64, 1.07, 0.00, 0.00, 0.00,
     0.00, 5.88, 7.62, 7.63, 3.66, 0.00, 0.00, 0.00, 0.00, 4.27, 7.55, 7.62, 3.81, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
     6.79, 5.26, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 5.11, 6.63, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 3.89, 7.63, 0.00,
     0.00, 0.00, 0.00, 0.00, 0.00, 2.28, 7.17],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.83, 2.29, 1.83, 0.00, 0.00, 0.00, 0.00,
     4.35, 7.63, 7.63, 7.40, 0.46, 0.00, 0.00, 0.00, 6.87, 6.33, 7.62, 6.94, 0.00, 0.00, 0.00, 0.00, 5.41, 7.63, 7.62,
     7.63, 0.77, 0.00, 0.00, 0.00, 0.46, 4.27, 4.95, 7.63, 2.52, 0.00, 0.00, 0.00, 0.00, 0.00, 0.61, 7.63, 3.66, 0.00,
     0.00, 0.00, 0.00, 0.00, 0.00, 6.33, 3.81]
])

new_labels = model.predict(new_samples)

for i in range(len(new_labels)):
    if new_labels[i] == 0:
        print(0, end='')
    elif new_labels[i] == 1:
        print(9, end='')
    elif new_labels[i] == 2:
        print(2, end='')
    elif new_labels[i] == 3:
        print(1, end='')
    elif new_labels[i] == 4:
        print(6, end='')
    elif new_labels[i] == 5:
        print(8, end='')
    elif new_labels[i] == 6:
        print(4, end='')
    elif new_labels[i] == 7:
        print(5, end='')
    elif new_labels[i] == 8:
        print(7, end='')
    elif new_labels[i] == 9:
        print(3, end='')
