import numpy as np
import pandas as pd

from sklearn import tree
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont

from sklearn.model_selection import train_test_split

full_data = pd.read_csv('../data/starcraft.csv')
full_data = full_data.dropna(axis=0, how='any')

# full_data = full_data.drop(['TotalHours', 'Age'], axis=1)

full_data['LeagueIndex'] = np.where(full_data.LeagueIndex > 4, 1, 0)

train, test = train_test_split(full_data)

y_train = train['LeagueIndex']
x_train = train.drop(['LeagueIndex'], axis=1).values 
x_test = test.drop(['LeagueIndex'], axis=1).values
y_test = test['LeagueIndex']

decision_tree = tree.DecisionTreeClassifier(max_depth = 2)
decision_tree.fit(x_train, y_train)

y_pred = decision_tree.predict(x_test)

submission = pd.DataFrame({
        "GameID": test['GameID'],
        "LeagueIndex": y_pred
    })
submission.to_csv('submission.csv', index=False)

with open("tree1.dot", 'w') as f:
     f = tree.export_graphviz(decision_tree,
                              out_file = f,
                              max_depth = 5,
                              impurity = True,
                              feature_names = list(train.drop(['LeagueIndex'], axis=1)),
                              class_names = ['0', '1'],
                              rounded = True,
                              filled= True )

check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])

img = Image.open("tree1.png")
draw = ImageDraw.Draw(img)
font = ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf', 26)
draw.text((10, 0),
          'test',
          (0,0,255),
          font=font)
img.save('sample-out.png')
PImage("sample-out.png")

print(decision_tree.score(x_test, y_test))

# http://blog.kaggle.com/2017/01/23/a-kaggle-master-explains-gradient-boosting/

# ----------------

from scipy.stats import randint as sp_randint

creator.create("FitnessMin", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

search_space = {"max_depth": sp_randint(1, 11),
                "max_features": sp_randint(1, 11),
                "criterion": ["gini", "entropy"]}

def main():
  # population = [creator.Individual(x) for x in (numpy.random.uniform(0, 1, (MU, N)))]
  population = [creator.Individual(x) for x in map(numpy.random.rand, search_space)]

  for ind in population:
      ind.fitness.values = toolbox.evaluate(ind)