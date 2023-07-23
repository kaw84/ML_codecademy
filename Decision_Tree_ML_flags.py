import codecademylib3_seaborn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

#load data into new variable
flags = pd.read_csv('flags.csv', header = 0)

#new variable
labels = flags['Landmass']
data = flags[['Red', 'Green', 'Blue', 'Gold', 'White', 'Black', 'Orange', 'Circles', 'Crosses', 'Saltires', 'Quarters', 'Sunstars', 'Crescent', 'Triangle']]

#splitting data
train_data, test_data, train_labels, test_labels = train_test_split(data, labels)

#list for plot
scores = []

for i in range(1,21):
  #creating a tree
  tree = DecisionTreeClassifier(random_state = 1, max_depth = i)

  #training the tree
  tree.fit(train_data, train_labels)

  #checking accuracy
  scores.append(tree.score(test_data, test_labels))

#creating a plot
plt.plot(range(1,21), scores)
plt.show()
