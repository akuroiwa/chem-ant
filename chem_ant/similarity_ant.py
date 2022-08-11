"""
This is based on the code of
`deap/examples/gp/ant.py <https://github.com/DEAP/deap/blob/master/examples/gp/ant.py>`__.
"""

import copy
import random

import numpy

from functools import partial

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

from copy import deepcopy
# from mcts import mcts, treeNode
try:
    from mcts_solver.mcts_solver import AntLionTreeNode, AntLionMcts
except ImportError:
    from mcts_solver import AntLionTreeNode, AntLionMcts

try:
    from .similarity_mcts import SimilarityState
except ImportError:
    from similarity_mcts import SimilarityState
import math
# import time
import os

import argparse
import pandas as pd
# from collections import deque
# import sys
# from concurrent.futures.process import ProcessPoolExecutor


# def progn(*args):
#     with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
#         for arg in args:
#             executor.submit(arg)

def progn(*args):
    for arg in args:
        arg()

def prog2(out1, out2):
    return partial(progn,out1,out2)

def prog3(out1, out2, out3):
    return partial(progn,out1,out2,out3)

def if_then_else(condition, out1, out2):
    out1() if condition() else out2()

def if_nest(condition, out1, out2, out3):
    out1() if condition() == 2 else out2() if condition() == 1 else out3()

class SimilarityAntSimulator(object):

    def __init__(self):
        self.previous_eaten = 0
        self.eaten = 0
        self.routine = None
        self.previous_length = float("inf")
        self.numVisits = 0
        self.improvement = 0
        self.shortcut = 0
        self.result = 0
        self.pruning = 0
        # self.initialState = SimilarityState(jewel, vera, loop)
        # self.mcts_instance = AntMcts(iterationLimit=5)

    def _reset(self):
        self.previous_eaten = 0
        self.eaten = 0
        self.previous_length = float("inf")
        self.numVisits = 0
        self.improvement = 0
        self.shortcut = 0
        self.result = 0
        self.pruning = 0

    @property
    def get_prediction(self):
        bestChild = self.mcts_instance.getBestChild(self.root, 0)
        action=(action for action, node in self.root.children.items() if node is bestChild).__next__()
        return action

    def set_loop(self, jewel, vera, double_clue, loop, experiment, file_name, json, verbose):
        self.initialState = SimilarityState(jewel, vera,
                                   loop, experiment, file_name, json, verbose)
        self.mcts_instance = AntMcts(iterationLimit=5)
        self.initialState.setPrevious(double_clue)
        # self.mcts_instance = AntMcts(iterationLimit=5)
        # self.root = AntTreeNode(self.initialState, None)
        self.root = AntLionTreeNode(self.initialState, None)

    def set_dl(self, regression=False):
        self.mcts_instance.dl = True
        # self.mcts_instance.regression = True
        self.mcts_instance.regression = regression
        if regression:
            try:
                from chem_classification.similarity_classification import SimilarityRegression
            except ImportError:
                from similarity_classification import SimilarityRegression
            self.mcts_instance.classification = SimilarityRegression()
        else:
            try:
                from chem_classification.similarity_classification import SimilarityClassification
            except ImportError:
                from similarity_classification import SimilarityClassification
            self.mcts_instance.classification = SimilarityClassification()

    def selectNode_1(self):
        node = self.mcts_instance.selectNode_num(self.root, 1 / math.sqrt(1))
        self._executeRound(node, 3)

    def selectNode(self):
        node = self.mcts_instance.selectNode(self.root)
        self._executeRound(node, 2)

    def selectNode_3(self):
        node = self.mcts_instance.selectNode_num(self.root, 1 / math.sqrt(3))
        self._executeRound(node, 3)

    def selectNode_4(self):
        node = self.mcts_instance.selectNode_num(self.root, 1 / math.sqrt(4))
        self._executeRound(node, 4)

    def selectNode_5(self):
        node = self.mcts_instance.selectNode_num(self.root, 1 / math.sqrt(5))
        self._executeRound(node, 5)

    def selectNode_6(self):
        node = self.mcts_instance.selectNode_num(self.root, 1 / math.sqrt(6))
        self._executeRound(node, 6)

    def selectNode_7(self):
        node = self.mcts_instance.selectNode_num(self.root, 1 / math.sqrt(7))
        self._executeRound(node, 7)

    def selectNode_8(self):
        node = self.mcts_instance.selectNode_num(self.root, 1 / math.sqrt(8))
        self._executeRound(node, 8)

    def selectNode_9(self):
        node = self.mcts_instance.selectNode_num(self.root, 1 / math.sqrt(9))
        self._executeRound(node, 9)

    def _executeRound(self, node, sqrt_num):
        # reward = self.mcts_instance.rollout(node.state)
        reward = self.mcts_instance.mctsSolver(node)
        length = len(node.state.hercule)
        self.mcts_instance.backpropogate(node, reward)
        self.numVisits = node.numVisits
        explorationValue = 1 / math.sqrt(sqrt_num)
        self.eaten = node.parent.state.getCurrentPlayer() * node.totalReward / node.numVisits + explorationValue * math.sqrt(
            2 * math.log(self.root.numVisits) / node.numVisits)
        self.improvement = 2 if self.eaten > self.previous_eaten else 1 if self.eaten == self.previous_eaten else 0
        self.previous_eaten = self.eaten
        self.shortcut = 2 if length > self.previous_length else 1 if length == self.previous_length else 0
        self.previous_length = length
        self.result = 2 if node.state.getReward() == 1 else 1 if node.state.getReward() == -1 else 0
        self.pruning = 2 if self.eaten == float("inf") else 1 if self.eaten == float("-inf") else 0

    def sense_improvement(self):
        return self.improvement

    def if_improvement(self, out1, out2, out3):
        return partial(if_nest, self.sense_improvement, out1, out2, out3)

    def sense_shortcut(self):
        return self.shortcut

    def if_shortcut(self, out1, out2, out3):
        return partial(if_nest, self.sense_shortcut, out1, out2, out3)

    def sense_result(self):
        return self.result

    def if_result(self, out1, out2, out3):
        return partial(if_nest, self.sense_result, out1, out2, out3)

    def sense_pruning(self):
        return self.pruning

    def if_pruning(self, out1, out2, out3):
        return partial(if_nest, self.sense_pruning, out1, out2, out3)

    def run(self,routine):
        self._reset()
        routine()


class AntMcts(AntLionMcts):

    def dl_method(self, bestChild):
        prediction, raw_outputs = self.classification.predict_smiles_pair(bestChild.state.jewel, ' '.join(bestChild.state.hercule))
        if self.regression:
            if prediction <= 0.5:
                dl_prediction = 0
            elif prediction > 0.5:
                dl_prediction = 1
            else:
                dl_prediction = -1
        else:
            if prediction == 2:
                dl_prediction = 1
            elif prediction == 1:
                dl_prediction = -1
            else:
                dl_prediction = 0
        reward = bestChild.state.getCurrentPlayer() * -dl_prediction


ant = SimilarityAntSimulator()

pset = gp.PrimitiveSet("MAIN", 0)
pset.addPrimitive(prog2, 2)
pset.addPrimitive(prog3, 3)
pset.addPrimitive(ant.if_improvement, 3)
pset.addPrimitive(ant.if_shortcut, 3)
pset.addPrimitive(ant.if_result, 3)
pset.addPrimitive(ant.if_pruning, 3)
pset.addTerminal(ant.selectNode_1)
pset.addTerminal(ant.selectNode)
pset.addTerminal(ant.selectNode_3)
pset.addTerminal(ant.selectNode_4)
pset.addTerminal(ant.selectNode_5)
pset.addTerminal(ant.selectNode_6)
pset.addTerminal(ant.selectNode_7)
pset.addTerminal(ant.selectNode_8)
pset.addTerminal(ant.selectNode_9)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("expr_init", gp.genGrow, pset=pset, min_=1, max_=2)

# Structure initializers
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_init)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalArtificialAnt(individual):
    # Transform the tree expression to functionnal Python code
    routine = gp.compile(individual, pset)
    ant.run(routine)
    return ant.eaten,

toolbox.register("evaluate", evalArtificialAnt)
toolbox.register("select", tools.selTournament, tournsize=7)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

def main(jewel, vera, double_clue, loop=1, experiment=3, file_name="generated_smiles.csv", population=500, generation=15, dl=False, regression=False, json=False, verbose=False):
    random.seed(69)
    ant.set_loop(jewel, vera, double_clue, loop, experiment, file_name, json, verbose)
    if dl:
    # if dl or regression:
        ant.set_dl()
    elif regression:
        ant.set_dl(regression)
    pop = toolbox.population(n=population)
    hof = tools.HallOfFame(100)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    with numpy.errstate(invalid='ignore'):
        algorithms.eaSimple(pop, toolbox, 0.5, 0.2, generation, stats, halloffame=hof)

    move = ant.get_prediction
    print('\nBest choice:\n', move)

    best_ind = tools.selBest(pop, 1)[0]
    print("\nBest individual is %s, %s" % (best_ind, best_ind.fitness.values))

    # return pop, hof, stats
    return pop, hof, stats, move

def console_script():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-t", "--target", dest='target', default=None, type=str, help="Target smile.")
    parser.add_argument("-m", "--molecule", dest='molecule', default=None, type=str, help="List of smiles from which the fragments are made.", nargs='+')
    # parser.add_argument("-p", "--previous", dest='previous', default=None, type=str, help="List of previously selected smiles.", nargs='+')
    parser.add_argument("-l", "--loop", dest='loop', default=1, type=int, help="For loop.  Default is 1.")
    parser.add_argument("-n", "--population", dest='population', default=500, type=int, help="Population size.  Default is 500.")
    parser.add_argument("-g", "--generation", dest='generation', default=15, type=int, help="The number of generation.  Default is 15")
    parser.add_argument("-i", "--include", dest='include', action='store_true', help="Include target smiles in list of smiles.")
    parser.add_argument("-e", "--experiment", dest='experiment', default=3, type=int, help="How many times you want to generate molecules in getReward process.  Default is 3.")
    parser.add_argument("-b", "--generate", dest='generate', default=10, type=int, help="Numbers of molecule generation.  Default is 10.")
    parser.add_argument("-s", "--select", dest='select', default=None, type=int, help="Select list of smiles from generated_smiles.csv.")
    parser.add_argument("-p", "--path", dest="path", default="gen_smiles", type=str, help="Directory where you want to save.  Default is gen_smiles.")
    parser.add_argument("-f", "--file", dest="file_name", default="generated_smiles.csv", type=str, help="File name.  Default is generated_smiles.csv.")
    parser.add_argument("-d", "--deep-learning", dest='dl', action='store_true', help="With deep learning.")
    parser.add_argument("-r", "--rgression", dest="regression", action="store_true", help="Use regression model.")
    parser.add_argument("-j", "--json", dest="json", action="store_true", help="Output json file for similarity_classification.py.")
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", help="Verbose mode.")
    args = parser.parse_args()

    if args.target == None:
        # jewel = "C1CCNC(C1)C(C2=CC(=NC3=C2C=CC=C3C(F)(F)F)C(F)(F)F)O"
        # jewel = "CC1([C@@H]2[C@H]1[C@H](N(C2)C(=O)[C@H](C(C)(C)C)NC(=O)C(F)(F)F)C(=O)N[C@@H](C[C@@H]3CCNC3=O)C#N)C"
        # Nirmatrelvir
        jewel = "CC1(C2C1C(N(C2)C(=O)C(C(C)(C)C)NC(=O)C(F)(F)F)C(=O)NC(CC3CCNC3=O)C#N)C"
    else:
        jewel = args.target
    if args.select:
        vera = pd.read_csv('generated_smiles.csv', header=0, usecols=[1], nrows=args.select).squeeze().values.tolist()
    elif args.molecule:
        vera = args.molecule
    else:
        vera = pd.read_csv('smiles.csv', header=None, usecols=[2]).squeeze().values.tolist()

    # if args.molecule == None:
    #     vera = pd.read_csv('smiles.csv', header=None, usecols=[2]).squeeze().values.tolist()
    # else:
    #     vera = args.molecule

    # initialState = SimilarityState(jewel, vera, loop=args.loop)
    # mcts = mcts(iterationLimit=5)

    double_clue = set()
    loop = args.loop
    experiment = args.experiment
    population = args.population
    generation = args.generation
    os.makedirs(args.path, exist_ok=True)
    file_name = os.path.join(args.path, args.file_name)
    dl = args.dl
    regression = args.regression
    json = args.json
    verbose = args.verbose
    for i in range(args.loop):
        # initialState.setPrevious(double_clue)
        # action = mcts.search(initialState=initialState)
        pop, hof, stats, action = main(jewel, vera,
                                       double_clue, loop, experiment, file_name, population, generation, dl, regression, json, verbose)
        double_clue.add(action)
        # loop -= 1

    if args.include:
        double_clue.add(args.target)
    print("Material candidates: {}".format(double_clue))
    generated_mols, smiles, morgan_fps, little_gray_cells, lipinski = ant.initialState.genMols(jewel, double_clue, args.generate, file_name)
    if verbose:
        print("Generated smiles: {}".format(smiles))

    # gen_df = pd.DataFrame({"smiles": smiles, "dice_similarity": little_gray_cells})
    # gen_df.sort_values("dice_similarity", inplace=True, ascending=False)
    # gen_df.reset_index(drop=True).to_csv("generated_smiles.csv")

if __name__ == "__main__":
    console_script()
