import random
import numpy as np
# np.random.seed(0)
# random.seed(0)

'''
Main file. From here I call all the relevant functions that allow me to test my
algorithm, including obtaining the graph Laplacian, learning an optimal policy
given a reward function, and plotting options and basis functions.

Author: Marlos C. Machado
'''
import sys
import math
import warnings
import numpy as np
import matplotlib.pylab as plt

from learning import Learning
from drawing import Plotter
from utils import Utils
from utils import ArgsParser
from environment import GridWorld
from mdpStats import MDPStats

from qlearning import QLearning


def discoverOptions(env, epsilon, discoverNegation):
    # I'll need this when computing the expected number of steps:
    options = []
    actionSetPerOption = []

    # Computing the Combinatorial Laplacian
    W = env.getAdjacencyMatrix()
    numStates = env.getNumStates()
    D = np.zeros((numStates, numStates))

    # Obtaining the Valency Matrix
    diag = np.sum(W, axis=0)

    # Making sure our final matrix will be full rank
    diag = np.clip(diag, 1.0, np.inf)
    D2 = np.diag(diag)

    # Normalized Laplacian
    L = D - W
    D2[D2 != 0] = np.power(D2[D2 != 0], -0.5)
    expD = D2
    normalizedL = expD.dot(L).dot(expD)

    # Eigendecomposition
    # IMPORTANT: The eigenvectors are in columns
    eigenvalues, eigenvectors = np.linalg.eig(normalizedL)
    # I need to sort the eigenvalues and eigenvectors
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # If I decide to use both directions of the eigenvector, I do it here.
    # It is easier to just change the list eigenvector, even though it may
    # not be the most efficient solution. The rest of the code remains the same.
    if discoverNegation:
        oldEigenvalues = eigenvalues
        oldEigenvectors = eigenvectors.T
        eigenvalues = []
        eigenvectors = []
        for i in range(len(oldEigenvectors)):
            eigenvalues.append(oldEigenvalues[i])
            eigenvalues.append(oldEigenvalues[i])
            eigenvectors.append(oldEigenvectors[i])
            eigenvectors.append(-1 * oldEigenvectors[i])

        eigenvalues = np.asarray(eigenvalues)
        eigenvectors = np.asarray(eigenvectors).T

    # if plotGraphs:
    #     # Plotting all the basis
    #     plot = Plotter(outputPath, env)
    #     plot.plotBasisFunctions(eigenvalues, eigenvectors)

    # Now I will define a reward function and solve the MDP for it
    # I iterate over the columns, not rows. I can index by 0 here.
    print("Solving for eigenvector #", end=' ')
    vecs = eigenvectors.T.copy()
    np.random.shuffle(vecs)

    for idx, eigenvector in enumerate(vecs):
        print(idx, end=' ')
        polIter = Learning(0.9, env, augmentActionSet=True)
        env.defineRewardFunction(eigenvector)
        V, pi = polIter.solvePolicyIteration()

        # Now I will eliminate any actions that may give us a small improvement.
        # This is where the epsilon parameter is important. If it is not set all
        # it will never be considered, since I set it to a very small value
        for j in range(len(V)):
            if V[j] < epsilon:
                pi[j] = len(env.getActionSet())

        options.append(pi[0:numStates])
        optionsActionSet = env.getActionSet()
        optionsActionSet.append('terminate')
        actionSetPerOption.append(optionsActionSet)

    print("\n")
    # I need to do this after I'm done with the PVFs:
    env.defineRewardFunction(None)
    env.reset()
    return options, actionSetPerOption


def policyEvaluation(env):
    ''' Simple test for policy evaluation '''

    pi = numStates * [[0.25, 0.25, 0.25, 0.25]]
    actionSet = env.getActionSet()

    # This solution is slower and it does not work for gamma = 1
    # polEval = Learning(0.9999, env, augmentActionSet=False)
    # expectation = polEval.solvePolicyEvaluation(pi)

    bellman = Learning(1, env, augmentActionSet=False)
    expectation = bellman.solveBellmanEquations(pi, actionSet, None)

    for i in range(len(expectation) - 1):
        sys.stdout.write(str(expectation[i]) + '\t')
        if (i + 1) % env.numCols == 0:
            print()
    print()


def policyIteration(env):
    ''' Simple test for policy iteration '''

    polIter = Learning(0.9, env, augmentActionSet=False)
    V, pi = polIter.solvePolicyIteration()

    # I'll assign the goal as the termination action
    pi[env.getGoalState()] = 4

    # Now we just plot the learned value function and the obtained policy
    plot = Plotter(outputPath, env)
    plot.plotValueFunction(V[0:numStates], 'goal_')
    plot.plotPolicy(pi[0:numStates], 'goal_')


def optionDiscoveryThroughPVFs(env, epsilon, verbose, discoverNegation):
    ''' Simple test for option discovery through proto-value functions. '''
    options, actionSetPerOption = discoverOptions(env,
                                                  epsilon=epsilon, verbose=verbose,
                                                  discoverNegation=discoverNegation, plotGraphs=verbose)

    # I convert the np.array of options into a list and print it.
    # This is useful if one wants to use this data in a different script.
    if verbose:
        print()
        print('Information about discovered options:')
        for i in range(len(options)):
            options[i] = options[i].tolist()
        print('numRows = ', env.getGridDimensions()[0])
        print('numCols = ', env.getGridDimensions()[1])
        print('env = ', env.matrixMDP.flatten().tolist())
        print('options = ', options)


def getExpectedNumberOfStepsFromOption(env, eps, verbose, discoverNegation, loadedOptions=None):
    # We first discover all options
    options = None
    actionSetPerOption = None
    actionSet = env.getActionSet()

    if loadedOptions == None:
        if verbose:
            options, actionSetPerOption = discoverOptions(env, eps, verbose,
                                                          discoverNegation, plotGraphs=True)
        else:
            options, actionSetPerOption = discoverOptions(env, eps, verbose,
                                                          discoverNegation, plotGraphs=False)
    else:
        options = loadedOptions
        actionSetPerOption = []
        for i in range(len(loadedOptions)):
            tempActionSet = env.getActionSet()
            tempActionSet.append('terminate')
            actionSetPerOption.append(tempActionSet)

    # Now I add all options to my action set. Later we decide which ones to use.
    for i in range(len(options)):
        actionSet.append(options[i])

    if loadedOptions == None:
        if discoverNegation:
            numOptions = 2 * env.getNumStates()
        else:
            numOptions = env.getNumStates()
    else:
        numOptions = len(loadedOptions)

    if discoverNegation:
        for i in range(numOptions / 2):
            listToPrint = stats.getAvgNumStepsBetweenEveryPoint(actionSet,
                                                                actionSetPerOption, verbose, initOption=i * 2,
                                                                numOptionsToConsider=2)
            myFormattedList = ['%.2f' % elem for elem in listToPrint]
            print('Random, Option ' + str(i + 1) + ': ' + str(myFormattedList))
    else:
        for i in range(numOptions):
            listToPrint = stats.getAvgNumStepsBetweenEveryPoint(actionSet,
                                                                actionSetPerOption, verbose, initOption=i,
                                                                numOptionsToConsider=1)
            myFormattedList = ['%.2f' % elem for elem in listToPrint]
            print('Random, Option ' + str(i + 1) + ': ' + str(myFormattedList))

    listToPrint = stats.getAvgNumStepsBetweenEveryPoint(actionSet,
                                                        actionSetPerOption, verbose, initOption=0,
                                                        numOptionsToConsider=numOptions)
    myFormattedList = ['%.2f' % elem for elem in listToPrint]
    print(myFormattedList)


def qLearningWithOptions(env, alpha, gamma, options_eps, epsilon, maxLengthEp, nEpisodes, verbose, useNegation,
                         genericNumOptionsToEvaluate, loadedOptions=None):
    # We first discover all options
    options = None
    actionSetPerOption = None

    if not loadedOptions:
        options, actionSetPerOption = discoverOptions(env, options_eps, useNegation)
    else:
        options = loadedOptions
        actionSetPerOption = []

        for i in range(len(loadedOptions)):
            tempActionSet = env.getActionSet()
            tempActionSet.append('terminate')
            actionSetPerOption.append(tempActionSet)

    returns_eval = []
    returns_learn = []
    # Now I add all options to my action set. Later we decide which ones to use.

    # genericNumOptionsToEvaluate = [1, 2, 4, 32, 64, 128, 256]
    totalOptionsToUse = []
    maxNumOptions = 0
    if useNegation and loadedOptions is None:
        maxNumOptions = int(len(options) / 2)
    else:
        maxNumOptions = len(options)

    numOptionsToUse = 5

    returns_eval.append([])
    returns_learn.append([])

    print('Using', numOptionsToUse, 'options')

    # returns_eval[idx].append([])
    # returns_learn[idx].append([])
    actionSet = env.getActionSet() + options[:numOptionsToUse]
    num_primitives = len(actionSet) - numOptionsToUse

    if useNegation and loadedOptions is None:
        numOptions = 2 * numOptionsToUse
    else:
        numOptions = numOptionsToUse

    learner = QLearning(alpha=alpha, gamma=gamma, epsilon=epsilon, environment=env, seed=1,
                        useOnlyPrimActions=False,
                        actionSet=actionSet, actionSetPerOption=actionSetPerOption)

    candidate_options = [i for i in range(numOptionsToUse)]
    for incumbent_idx in range(numOptionsToUse, len(options)):
        for i in range(nEpisodes):
            cum_reward = learner.learnOneEpisode(timestepLimit=maxLengthEp)
            cum_reward = learner.evaluateOneEpisode(eps=0.01, timestepLimit=maxLengthEp)

        performances = learner.Q.sum(axis=0)
        opt_performance = performances[num_primitives:]
        primitive_performance = performances[:num_primitives]
        print("performance: primitive-", primitive_performance, "option-", opt_performance)
        performers = opt_performance.argsort()
        worst_performer_idx = performers[0]
        # TODO: think
        #
        #  learner.Q[:, num_primitives + worst_performer_idx] = 0.5 * learner.Q.mean(axis=1)

        # hype driven exploration
        # learner.Q[:, num_primitives + worst_performer_idx] = learner.Q.max(axis=1).copy()

        # lobotomy
        learner.Q.fill(0)

        learner.actionSet[num_primitives + worst_performer_idx] = options[incumbent_idx]

        worst_performer = candidate_options[worst_performer_idx]
        candidate_options[worst_performer_idx] = incumbent_idx
        print("replaced {}(score {})".format(worst_performer, opt_performance[worst_performer_idx]))
    print("best options found", sorted(candidate_options))
    return candidate_options


if __name__ == "__main__":
    # Read input arguments
    args = ArgsParser.readInputArgs()
    epsilon = args.epsilon
    inputMDP = args.input
    outputPath = args.output
    bothDirections = args.both
    num_seeds = args.num_seeds
    max_length_episode = args.max_length_ep
    num_episodes = 200

    train_env = GridWorld(path=inputMDP, useNegativeRewards=False, env_id=0)
    # numStates = train_env.getNumStates()
    # numRows, numCols = train_env.getGridDimensions()
    # loadedOptions = None

    # Discover options
    # optionDiscoveryThroughPVFs(env=env, epsilon=epsilon, verbose=True, discoverNegation=bothDirections)

    # Solve for a given goal w/ primitive actions (q-learning)
    numOptionsDiscoveredToConsider = 128
    selected_options = qLearningWithOptions(
        env=train_env, alpha=0.1, gamma=0.9, options_eps=0.0, epsilon=1.0,
        maxLengthEp=max_length_episode, nEpisodes=num_episodes, verbose=False, useNegation=bothDirections,
        genericNumOptionsToEvaluate=[numOptionsDiscoveredToConsider], )

    # test_env = GridWorld(path=inputMDP, useNegativeRewards=False, env_id=1)
    # num_iterations = qLearningWithOptions(
    #     env=test_env, alpha=0.1, gamma=0.9, options_eps=0.0, epsilon=1.0, maxLengthEp=max_length_episode,
    #     nEpisodes=num_episodes, verbose=False, useNegation=bothDirections,
    #     genericNumOptionsToEvaluate=[numOptionsDiscoveredToConsider], loadedOptions=selected_options)
