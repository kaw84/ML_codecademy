import codecademylib3_seaborn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from svm_visualization import draw_boundary
from players import aaron_judge, jose_altuve, david_ortiz

fig, ax = plt.subplots()


#reassigning the data

def strike_zone(dataframe):
  dataframe['type'] = dataframe['type'].map({'S': 1, 'B': 0})

#drop nans
  dataframe = dataframe.dropna(subset = ['plate_x', 'plate_x', 'type'])

#scatter plot
  plt.scatter(x = dataframe['plate_x'], y = dataframe['plate_z'],c = dataframe['type'], cmap = plt.cm.coolwarm, alpha = 0.5)

  training_set, validation_set = train_test_split(dataframe, random_state = 1)


#object
  classifier = SVC(gamma = 3, C = 1)

#training
  classifier.fit(training_set[['plate_x', 'plate_z']], training_set['type'])

  #draw_boundary(ax, classifier)
  ax.set_xlim(-3,3)
  ax.set_ylim(-2,6)
  plt.show()

strike_zone(aaron_judge)
