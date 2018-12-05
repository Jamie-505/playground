# import numpy as np
# from pommerman.forward_model import ForwardModel
# import random
# from math import *
# from .. import constants
# from pommerman.agents.simple_agent import SimpleAgent
# import hashlib
# import argparse
#
# act = ForwardModel.act
# step = ForwardModel.step
#
# """
# A quick Monte Carlo Tree Search implementation.  For more details on MCTS see See http://pubs.doc.ic.ac.uk/survey-mcts-methods/survey-mcts-methods.pdf
#
# The State is just a game where you have NUM_TURNS and at turn i you can make
# a choice from [-2,2,3,-3]*i and this to to an accumulated value.  The goal is for the accumulated value to be as close to 0 as possible.
#
# The game is not very interesting but it allows one to study MCTS which is.  Some features
# of the example by design are that moves do not commute and early mistakes are more costly.
#
# In particular there are two models of best child that one can use
# """
#
#
# class MCTNode:
#
#     # TODO: use act with obs.enemies to get actions for step
#     # TODO: use step for simulations
#     # TODO: need actions, curr_board, curr_agents, curr_bombs, curr_items, curr_flames (probably parse board)
#
#     def __init__(self, agent_id, obs, action=None, parent=None):
#         self.action = action
#         self.agent_id = agent_id
#         self.children = []
#         self.obs = obs
#         self.parent = parent
#         self.untried_actions = list(range(6))
#         self.rewards = [0, 0, 0, 0]
#         self.reward_sum = 0
#         self.visits = 0
#         self.done = False
#
#     def select_child(self):
#         # select child node via UCB formula
#         selection = sorted(self.children, key=lambda c: c.reward_sum/c.visits + sqrt(2*log(self.visits) / c.visits))
#         return selection[-1]
#
#     def get_action(self):
#         return self.action
#
#     def expand(self, action, rewards, done):
#         self.untried_actions.remove(action)
#         child = MCTNode(action, rewards, parent=self)
#         child.set_final(done)
#         self.children.append(child)
#         return child
#
#     def corrected_rewards(self, rewards):
#         # TODO: needs to be more general (invert for not agent_id)
#         return [rewards[i] * (-1) if i != self.agent_id else rewards[i] for i in range(4)]
#
#     def update(self, rewards):
#         self.visits += 1
#         self.rewards = rewards
#         cr = self.corrected_rewards(rewards)
#         self.reward_sum += sum(cr)
#
#     def final(self):
#         return self.done
#
#     def set_final(self, done):
#         self.done = done
#
#     def __repr__(self):
#         return "[M:" + str(self.move) + " W/V:" + str(self.wins) + "/" + str(self.visits) + " U:" + str(
#             self.untriedMoves) + "]"
#
#     def tree_to_string(self, indent):
#         s = self.indent_string(indent) + str(self)
#         for c in self.children:
#             s += c.tree_to_string(indent + 1)
#         return s
#
#     @staticmethod
#     def indent_string(indent):
#         s = "\n"
#         for i in range(1, indent + 1):
#             s += "| "
#         return s
#
#     def children_to_string(self):
#         s = ""
#         for c in self.children:
#             s += str(c) + "\n"
#         return s
#
#
# def uct(agent, node, tree, verbose=False):
#     done = False
#     rewards = [0, 0, 0, 0]
#     assert agent.env.training_agent == agent.agent_id
#
#     state = str(agent.env.get_json_info)
#     tree[state] = node
#
#     if node.final:
#         return node.reward_sum
#     elif
#
#     # Select
#     while node.untried_actions == [] and not node.final():  # node is fully expanded and non-terminal
#         node = node.select_child()
#         action = node.get_action()
#         actions = agent.env.act(obs)
#         actions.insert(agent.agent_id, action)
#         obs, rewards, done, info = agent.env.step(actions)
#
#     # Expand
#     if node.untried_actions:
#         action = random.choice(node.untried_actions)
#         actions = agent.env.act(obs)
#         actions.insert(agent.agent_id, action)
#         obs, rewards, done, info = agent.env.step(actions)
#         node = node.expand(action, rewards, done)
#         tree[str(agent.env.get_json_info())] = node
#
#     # Simulate
#     steps = 0
#     while not done:
#         agent.env.render()
#
#         # ensure we are not called recursively
#         assert agent.env.training_agent == agent.agent_id
#         # make other agents act
#         actions = agent.env.act(obs)
#         # add my action to list of actions
#         actions.insert(agent.agent_id, action)
#         # step environment
#         obs, rewards, done, info = agent.env.step(actions)
#         assert agent == agent.env._agents[agent.agent_id]
#         steps += 1
#         print("Agent:", agent.agent_id, "Step:", steps, "Actions:", [constants.Action(a).name for a in actions],
#               "Rewards:", rewards, "Done:", done)
#
#     # Backpropgate
#     while node is not None:  # go all the way back to the root
#         node.update(rewards)
#         node = node.parent
#
#     # Output some information about the tree - can be omitted
#     if verbose:
#         print(rootnode.tree_to_string(0))
#     else:
#         print(rootnode.children_to_string())
#
#     return sorted(rootnode.childNodes, key=lambda c: c.visits)[
#                -1].get_action, tree  # return the move that was most visited
