def problem(individual, functions):
    return list(map(lambda x : x(individual), functions))

def accuracy(individual):
    pass

def processing_time(individual):
    pass

# if __name__ == "__main__":
#     print(problem([2,3], [sum, mul]))

# https://github.com/claesenm/optunity/blob/c5810d7aae2be06e6ab619f2e5ab294778cf6fb2/optunity/solvers/CMAES.py