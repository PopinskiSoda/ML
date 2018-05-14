import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
# from sklearn.metrics import accuracy_score

from scipy.stats import randint as sp_randint
from deap import creator, base, cma, tools

from time import time

full_data = pd.read_csv('./data/starcraft.csv')
full_data = full_data.dropna(axis=0, how='any')

full_data = full_data.drop(['TotalHours', 'Age'], axis=1)
full_data['LeagueIndex'] = np.where(full_data.LeagueIndex > 4, 1, 0)
train, test = train_test_split(full_data)

y_train = train['LeagueIndex']
x_train = train.drop(['LeagueIndex'], axis=1).values 
x_test = test.drop(['LeagueIndex'], axis=1).values
y_test = test['LeagueIndex']

decision_tree = tree.DecisionTreeClassifier(max_depth = 5)
decision_tree.fit(x_train, y_train)

y_pred = decision_tree.predict(x_test)

N = 2

MIN_BOUND = np.zeros(N) + 2
MAX_BOUND = np.zeros(N) + 15
EPS_BOUND = 2.e-5

def accuracy(ind):
    # print(int(round(ind[0])), int(round(ind[1])))
    decision_tree.set_params(max_depth=int(round(ind[0])), max_features=int(round(ind[1])))
    decision_tree.fit(x_train, y_train)
    # print(decision_tree.score(x_test, y_test))
    return decision_tree.score(x_test, y_test)

def processing_time(ind):
    decision_tree.set_params(max_depth=int(round(ind[0])), max_features=int(round(ind[1])))
    start = time()
    decision_tree.fit(x_train, y_train)
    return time() - start

def test_func(ind):
    return accuracy(ind), processing_time(ind)

def distance(feasible_ind, original_ind):
    return sum((f - o)**2 for f, o in zip(feasible_ind, original_ind))

def closest_feasible(individual):
    feasible_ind = np.array(individual)
    feasible_ind = np.maximum(MAX_BOUND, feasible_ind)
    feasible_ind = np.minimum(MIN_BOUND, feasible_ind)
    return feasible_ind

def valid(individual):
    if any(individual < MIN_BOUND) or any(individual > MAX_BOUND):
        return False
    return True

def close_valid(individual):
    if any(individual < MIN_BOUND-EPS_BOUND) or any(individual > MAX_BOUND+EPS_BOUND):
        return False
    return True



toolbox = base.Toolbox()
toolbox.register("evaluate", test_func)
toolbox.decorate("evaluate", tools.ClosestValidPenalty(valid, closest_feasible, 1.0e+6, distance))

creator.create("FitnessMin", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

search_space = {"max_depth": np.arange(2, 15),
                "max_features": np.arange(2, 15)}
                # "criterion": [1, 2]}
                # "criterion": ["gini", "entropy"]}

search_space_values = list(search_space.values())

def get_random_parameters():
    yield list(map(np.random.choice, search_space_values))

def main():
    NGEN = 100
    MU, LAMBDA = 10, 10
    verbose = True
    # create_plot = True

    population = []
    for i in range(0, 10):
        population.append(creator.Individual(*get_random_parameters()))

    # print(population)
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)
    # hof = tools.HallOfFame(1)

    strategy = cma.StrategyMultiObjective(population, sigma=1.0, mu=MU, lambda_=LAMBDA)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    fitness_history = []
    
    for gen in range(NGEN):
        population = toolbox.generate()

        # Evaluate
        fitnesses = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
            fitness_history.append(fit)
        
        toolbox.update(population)
        
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(population), **record)
        if verbose:
            print(logbook.stream)

    # print(population)
    # print(hof[0])
    # print(population, list(map(test_func, population)))
    for ind in population:
        acc, pr_time = test_func(ind)
        print("ind: {0} accuracy: {1} time: {2}".format(ind, acc, pr_time))

if __name__ == "__main__":
    main()
