# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best


        if len(bestIndices) > 1 and  legalMoves[chosenIndex] == "Stop":
            k = chosenIndex
            for i in bestIndices :
                if i != k :
                    chosenIndex = i

        "Add more of your code here if you want to"
        #print(scores, bestIndices, legalMoves[chosenIndex], chosenIndex)
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # take food and the position of ghosts into consideration with different weight
        # nearer ghosts --bad;


        minGhostDist = -1
        ngCost = 0
        for ghostState in newGhostStates:
            dist = manhattanDistance(newPos, ghostState.getPosition())
            if dist < minGhostDist or minGhostDist == -1:
                minGhostDist = dist
        if minGhostDist == 1:
            ngCost += -3000
        elif minGhostDist == 0:
            ngCost += -6000
        elif minGhostDist >= 18:
            ngCost += 100
        else:
            ngCost = 0

        # more food --good;
        foodNum = len(newFood.asList())
        fnCost = -foodNum * 30
        # more foodCost --bad;
        minFoodDist = -1
        fcCost = 0
        for food in newFood.asList():
            foodDist = manhattanDistance(newPos, food)
            if foodDist < minFoodDist or minFoodDist == -1:
                minFoodDist = foodDist
        if minFoodDist == -1 :
            fcCost = 0
        else :
            fcCost = -minFoodDist
        # newScaredTimes --good
        scaredTimes = 0
        scared = False
        for ghostState in newGhostStates:
            if ghostState.scaredTimer > 35:
                scaredTimes += 10
                scared = True
            elif ghostState.scaredTimer >= 30 and ghostState.scaredTimer <= 35:
                scaredTimes += 5
                scared = True
            elif ghostState.scaredTimer > 0 and ghostState.scaredTimer <= 30:
                scaredTimes += 1
                scared = True
        stCost = 10 * scaredTimes
        if scared and minGhostDist <= 1: ngCost += 1000
        #print(stCost, fcCost, fnCost, ngCost,minFoodDist)

        return stCost + fcCost + fnCost + ngCost

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        # decide which action is best for Pacman
        # Pacman --maxValue Ghosts --minValue
        # every move of Pacman follows all ghosts' movement
        PacIndex = 0
        maxValue = float("-inf")
        for action in gameState.getLegalActions(PacIndex):
            nextState = gameState.generateSuccessor(PacIndex, action)
            nextValue = self.evaluate(nextState, 0, 1)
            if nextValue >= maxValue:
                maxValue = nextValue
                objAction = action
        return objAction
        util.raiseNotDefined()

    def evaluate(self, gameState: GameState, depth, index):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        if index == 0:
            maxValue = float("-inf")
            # maxValue of Pacman and ghosts in the same depth
            for action in gameState.getLegalActions(index):
                maxValue = max(maxValue, self.evaluate(gameState.generateSuccessor(index, action), depth, index + 1))
            return maxValue
        else:
            minValue = float("inf")
            # if the last ghost, consider the next level
            for action in gameState.getLegalActions(index) :
                if index == gameState.getNumAgents() - 1:
                    minValue = min(minValue, self.evaluate(gameState.generateSuccessor(index, action), depth + 1, 0))
                else:
                    minValue = min(minValue, self.evaluate(gameState.generateSuccessor(index, action), depth, index + 1))
            return minValue




class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        PacIndex = 0
        maxValue = float("-inf")
        alpha = float("-inf") # 当前最大化玩家已知最优值
        beta = float("inf") # 当前最小化玩家已知最优值
        objAction = Directions.STOP
        for action in gameState.getLegalActions(PacIndex):
            nextState = gameState.generateSuccessor(PacIndex, action)
            nextValue = self.evaluate(nextState, 0, 1, alpha, beta)
            if nextValue > maxValue:
                maxValue = nextValue
                objAction = action
            alpha = max(alpha, maxValue)
        return objAction
        util.raiseNotDefined()

    def evaluate(self, gameState, depth, index, alpha, beta):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        if index == 0:
            maxValue = float("-inf")
            # maxValue of Pacman and ghosts in the same depth
            for action in gameState.getLegalActions(index):
                maxValue = max(maxValue, self.evaluate(gameState.generateSuccessor(index, action), depth, index + 1, alpha, beta))
                if maxValue > beta:
                    return maxValue
                alpha = max(alpha, maxValue)
            return maxValue
        else :
            minValue = float("inf")
            # if the last ghost, consider the next level
            for action in gameState.getLegalActions(index):
                if index == gameState.getNumAgents() - 1:
                    minValue = min(minValue, self.evaluate(gameState.generateSuccessor(index, action), depth + 1, 0, alpha, beta))
                else:
                    minValue = min(minValue,
                                   self.evaluate(gameState.generateSuccessor(index, action), depth, index + 1, alpha, beta))
                if minValue < alpha:
                    return minValue
                beta = min(minValue, beta)
            return minValue

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        PacIndex = 0
        maxValue = float("-inf")
        for action in gameState.getLegalActions(PacIndex):
            nextState = gameState.generateSuccessor(PacIndex, action)
            nextValue = self.evaluate(nextState, 0, 1)
            if nextValue >= maxValue:
                maxValue = nextValue
                objAction = action
        return objAction
        util.raiseNotDefined()


    def evaluate(self, gameState: GameState, depth, index):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        if index == 0:
            maxValue = float("-inf")
            # maxValue of Pacman and ghosts in the same depth
            for action in gameState.getLegalActions(index):
                maxValue = max(maxValue, self.evaluate(gameState.generateSuccessor(index, action), depth, index + 1))
            return maxValue
        else:
            average = 0.0
            stateNum = len(gameState.getLegalActions(index))
            # if the last ghost, consider the next level
            for action in gameState.getLegalActions(index) :
                if index == gameState.getNumAgents() - 1:
                    average += self.evaluate(gameState.generateSuccessor(index, action), depth + 1, 0) / stateNum
                else:
                    average += self.evaluate(gameState.generateSuccessor(index, action), depth, index + 1) / stateNum
            return average


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    # Useful information you can extract from a GameState (pacman.py)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    Capsules = currentGameState.getCapsules()
    # take food and the position of ghosts into consideration with different weight
    # nearer ghosts --bad

    minGhostDist = -1
    ngCost = 0
    for ghostState in newGhostStates:
        dist = manhattanDistance(newPos, ghostState.getPosition())
        if dist < minGhostDist or minGhostDist == -1:
            minGhostDist = dist
    if minGhostDist == 1:
        ngCost += -3000
    elif minGhostDist == 0:
        ngCost += -6000
    elif minGhostDist >= 18:
        ngCost += 100
    else: ngCost = 0

    # more food --good;
    foodNum = len(newFood.asList())
    fnCost = -foodNum * 50
    # more foodCost --bad;
    minFoodDist = -1
    for food in newFood.asList():
        foodDist = manhattanDistance(newPos, food)
        if foodDist < minFoodDist or minFoodDist == -1:
            minFoodDist = foodDist
    if minFoodDist == -1:
        fcCost = 0
    else:
        fcCost = -minFoodDist
    # newScaredTimes --good
    scaredTimes = 0
    scared = False
    for ghostState in newGhostStates:
        if ghostState.scaredTimer > 35:
            scaredTimes += 10
            scared = True
        elif ghostState.scaredTimer >= 30 and ghostState.scaredTimer <= 35:
            scaredTimes += 5
            scared = True
        elif ghostState.scaredTimer > 0 and ghostState.scaredTimer <= 30:
            scaredTimes += 1
            scared = True
    stCost = 10 * scaredTimes
    if scared and minGhostDist <= 1: ngCost += 1000
    # print(stCost, fcCost, fnCost, ngCost,minFoodDist)
    #print(stCost, fcCost, fnCost, ngCost, cdCost, wallCost)


    return stCost + fcCost + fnCost + ngCost + currentGameState.getScore()

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
