# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp: mdp.MarkovDecisionProcess, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()
        # A Counter is a dict with default 0
        self.runValueIteration()
        
    def runValueIteration(self):
        """
          Run the value iteration algorithm. Note that in standard
          value iteration, V_k+1(...) depends on V_k(...)'s.
        """
        "*** YOUR CODE HERE ***"
        #BELLMAN UPDATE
        #V_K+1 = max_a sumof_s'(T(s, a, s')[R(s, a, s') + lambdaV_k(s')]  )
        states = self.mdp.getStates()
        for i in range(0, self.iterations):
            newVals = util.Counter()
            for state in states:
                if self.mdp.isTerminal(state):
                    continue
                actions = self.mdp.getPossibleActions(state)
                max = -100000
                for a in actions:
                    successors = self.mdp.getTransitionStatesAndProbs(state, a)
                    sum = 0
                    for succ in successors:
                        next, p = succ
                        reward = self.mdp.getReward(state, a, next)
                        sum += p * (reward + self.discount * self.values[next])
                    if sum > max:
                        max = sum
                newVals[state] = max
            
            self.values = newVals

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        #s' is successor
        #Q(s, a) = sumof_s'(T(s, a, s')[R(s, a, s') + lambdaV(s')])
        qval = 0
        successors = self.mdp.getTransitionStatesAndProbs(state, action)
        for succ in successors:
            next, p = succ
            reward = self.mdp.getReward(state, action, next)
            qval += p * (reward + self.discount * self.values[next])
        return qval
    
    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        #Find expected value of each action from state
        actions = self.mdp.getPossibleActions(state)
        if self.mdp.isTerminal(state):
            return None
        max = -100000
        best = None
        for a in actions:
            successors = self.mdp.getTransitionStatesAndProbs(state, a)
            expected = 0
            for succ in successors:
                next, p = succ
                expected += p * self.values[next]
            if expected > max:
                max = expected
                best = a        
        return best

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
