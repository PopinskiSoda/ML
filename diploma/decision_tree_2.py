import numpy as np
import pandas as pd
import re
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.offline as py
# py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from time import time
from scipy.stats import randint as sp_randint

import deap

full_data = pd.read_csv('./data/starcraft.csv')
# full_data = full_data[full_data.LeagueIndex != 8]
full_data = full_data.dropna(axis=0, how='any')

# Mapping data: 

# full_data['LeagueIndex'] = full_data['LeagueIndex']
full_data = full_data.drop(['TotalHours', 'Age'], axis=1)

full_data['LeagueIndex'] = np.where(full_data.LeagueIndex > 4, 1, 0)

train, test = train_test_split(full_data)

# full_data.head(10)

# full_data.isnull().values.any()

# train[['LeagueIndex', 'TotalHours', 'APM', 'HoursPerWeek', 'Age', 'AssignToHotkeys', 'TotalMapExplored', 'WorkersMade', 'ComplexAbilityUsed', 'ComplexUnitsMade']].groupby(['LeagueIndex'], as_index=False).agg(['mean', 'count'])

y_train = train['LeagueIndex']
x_train = train.drop(['LeagueIndex'], axis=1).values 
x_test = test.drop(['LeagueIndex'], axis=1).values
y_test = test['LeagueIndex']

decision_tree = tree.DecisionTreeClassifier(max_depth = 5)
decision_tree.fit(x_train, y_train)

y_pred = decision_tree.predict(x_test)

GameID = test['GameID']

submission = pd.DataFrame({
        "GameID": GameID,
        "LeagueIndex": y_pred
    })
submission.to_csv('submission.csv', index=False)

# Export our trained model as a .dot file
with open("tree1.dot", 'w') as f:
     f = tree.export_graphviz(decision_tree,
                              out_file = f,
                              max_depth = 5,
                              impurity = True,
                              feature_names = list(train.drop(['LeagueIndex'], axis=1)),
                              class_names = ['0', '1'],
                              rounded = True,
                              filled= True )
        
#Convert .dot to .png to allow display in web notebook
check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])

# Annotating chart with PIL
img = Image.open("tree1.png")
draw = ImageDraw.Draw(img)
font = ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf', 26)
draw.text((10, 0), # Drawing offset (position)
          '"Title <= 1.5" corresponds to "Mr." title', # Text to draw
          (0,0,255), # RGB desired color
          font=font) # ImageFont object with desired font
img.save('sample-out.png')
PImage("sample-out.png")

# specify parameters and distributions to sample from
param_dist = {"max_depth": sp_randint(1, 11),
              "max_features": sp_randint(1, 11),
              # "min_samples_split": sp_randint(2, 11),
              # "min_samples_leaf": sp_randint(1, 11),
              "criterion": ["gini", "entropy"]}

# run randomized search
n_iter_search = 12
random_search = RandomizedSearchCV(decision_tree, param_distributions=param_dist,
                                   n_iter=n_iter_search)

# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

start = time()
random_search.fit(x_test, y_test)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.cv_results_)

# use a full grid over all parameters
param_grid = {"max_depth": [3, 5, 10],
              "max_features": [1, 3, 10],
              # "min_samples_split": [2, 3, 10],
              # "min_samples_leaf": [1, 3, 10],
              "criterion": ["gini", "entropy"]}

# run grid search
grid_search = GridSearchCV(decision_tree, param_grid=param_grid)
start = time()
grid_search.fit(x_test, y_test)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)


# http://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html

# import numpy as np
# import scipy as sp
# import scipy.optimize

# def test_func(x):
#     return (x[0])**2+(x[1])**2+285

# def test_grad(x):
#     return [2*x[0],2*x[1]]

# myvar = sp.optimize.line_search(decision_tree,test_grad,np.array([1.8,1.8]),np.array([-1.,-1.]))
# print(myvar)

# print(decision_tree.get_params())
print(decision_tree)