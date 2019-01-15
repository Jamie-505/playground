from pommerman.forward_model import ForwardModel
import random
from math import *
from pommerman import constants
import numpy as np

act = ForwardModel.act
step = ForwardModel.step

"""
A quick Monte Carlo Tree Search implementation.
For more details on MCTS see See http://pubs.doc.ic.ac.uk/survey-mcts-methods/survey-mcts-methods.pdf

The State is just a game where you have NUM_TURNS and at turn i you can make
a choice from [-2,2,3,-3]*i and this to to an accumulated value.
The goal is for the accumulated value to be as close to 0 as possible.

The game is not very interesting but it allows one to study MCTS which is.  Some features 
of the example by design are that moves do not commute and early mistakes are more costly.  

In particular there are two models of best child that one can use 
"""


class MCTNode:

    def __init__(self, agent_id, action=0, parent=None):
        self.action = action
        # probability with which the action was sampled, defaults to 1/6
        self.probability = 1/6
        self.agent_id = agent_id
        self.children = []
        self.parent = parent
        self.untried_actions = list(range(6))
        self.rewards = [0, 0, 0, 0]
        self.reward_sum = 0
        self.visits = 0
        self.done = False

    # only gets called if fully expanded
    def select_child(self):
        # select child with exp3 formula

        p_distro, rewards, e_nu_omega = [], [], []
        # 1 - gamma
        pre = 5/6
        # |A| = 6 and gamma = 1/6
        nu = 1 / 36
        # get all rewards so the sum and max reward can be calculated
        for c in self.children:
            rewards.append(c.get_reward())
        max_r = np.amax(rewards)
        for c in self.children:
            e_nu_omega.append(exp(nu*(c.get_reward() - max_r)))
        sum_e_nu_omega = sum(e_nu_omega)
        for i in range(6):
            sigma = (pre * e_nu_omega[i] / sum_e_nu_omega) + nu
            p_distro.append(sigma)
            self.children[i].set_probability(sigma)

        return np.random.choice(self.children, p=p_distro)

    def get_action(self):
        return self.action

    def get_reward(self):
        return self.reward_sum

    def set_probability(self, p):
        self.probability = p

    def expand(self, action, done):
        self.untried_actions.remove(action)
        child = MCTNode(self.agent_id, action, parent=self)
        child.set_final(done)
        self.children.append(child)
        return child

    def corrected_rewards(self, rewards):
        return [rewards[i] * (-1) if i != self.agent_id else rewards[i] for i in range(4)]

    def update(self, rewards):
        self.visits += 1
        self.rewards = rewards
        corrected_rewards = self.corrected_rewards(rewards)
        self.reward_sum += sum(corrected_rewards)/self.probability

    def final(self):
        return self.done

    def set_final(self, done):
        self.done = done

    def __repr__(self):
        return "[Action:" + constants.Action(self.action).name + " W/V:" + str(self.reward_sum) + "/" \
               + str(self.visits)+ " Untried:" + ''.join([constants.Action(a).name for a in self.untried_actions]) + "]"

    def tree_to_string(self, indent):
        s = self.indent_string(indent) + str(self)
        for c in self.children:
            s += c.tree_to_string(indent + 1)
        return s

    @staticmethod
    def indent_string(indent):
        s = "\n"
        for i in range(1, indent + 1):
            s += "| "
        return s

    def children_to_string(self):
        s = ""
        for c in self.children:
            s += str(c) + "\n"
        return s


def uct(agent, root_state, tree, itermax, verbose=False):
    rewards = [0, 0, 0, 0]
    done = False

    assert agent.env.training_agent == agent.agent_id
    agent.env._init_game_state = root_state
    obs = agent.env.reset() # sets env to _init_game_state and return obs
    rootnode = MCTNode(agent.agent_id)

    for i in range(itermax):
        node = rootnode
        state = str(agent.env.get_json_info())
        tree[state] = node

        # Select
        while node.untried_actions == [] and not node.final():  # node is fully expanded and non-terminal
            node = node.select_child()
            action = node.get_action()
            actions = agent.env.act(obs)
            actions.insert(agent.agent_id, action)
            obs, rewards, done, info = agent.env.step(actions)

        # Expand
        if node.untried_actions:
            action = random.choice(node.untried_actions)
            actions = agent.env.act(obs)
            actions.insert(agent.agent_id, action)
            obs, rewards, done, info = agent.env.step(actions)
            node = node.expand(action, done)
            tree[str(agent.env.get_json_info())] = node

        # Simulate
        steps = 0
        print("\nSIMULATION BEGINS:")
        while not done:
            agent.env.render()

            # ensure we are not called recursively
            assert agent.env.training_agent == agent.agent_id
            action = random.choice(range(6))
            # make other agents act
            actions = agent.env.act(obs)
            # add my action to list of actions
            actions.insert(agent.agent_id, action)
            # step environment
            obs, rewards, done, info = agent.env.step(actions)
            assert agent == agent.env._agents[agent.agent_id]
            steps += 1
            print("Agent:", agent.agent_id, "Step:", steps, "Actions:", [constants.Action(a).name for a in actions],
                  "Rewards:", rewards, "Done:", done)

        # Backpropgate
        while node is not None:  # go all the way back to the root
            node.update(rewards)
            node = node.parent

        agent.env.reset()

    # Output some information about the tree - can be omitted
    if verbose:
        print(rootnode.tree_to_string(0))
    else:
        print(rootnode.children_to_string())

    # restore to _init_game_state, so "real" game can continue
    agent.env.set_json_info()

    #prepare final sampling
    prob_final_unscaled = []
    sum_visits = sum(list(map(lambda c: c.visits, rootnode.children)))
    for c in rootnode.children:
        prob_final_unscaled.append(max(0, (c.visits - (1/36) * sum_visits))/sum_visits)
    # scale probs to 1
    sum_probs = sum(prob_final_unscaled)
    probs_final = list(map(lambda p: p/sum_probs, prob_final_unscaled))
    chosen_action = np.random.choice(rootnode.children, p=probs_final).get_action()

    return chosen_action, tree  # return the move that was most visited
