import pommerman
from pommerman.agents.simple_agent import SimpleAgent
from pommerman.agents.random_agent import RandomAgent

from . import BaseAgent
from pommerman.mcts.mcts import UCT

class MctsAgent(BaseAgent):

  def __init__(self, agent_id=0, *args, **kwargs):
    super(MctsAgent, self).__init__(*args, **kwargs)
    self.agent_id = agent_id
    self.env = self.make_env()
    self.env.training_agent = self.agent_id
    self.reset_tree()

  def make_env(self):
    # idea to make indepent objects for planing
    agents = []
    for agent_id in range(4):
      if agent_id == self.agent_id:
        agents.append(self)
      else:
        agents.append(SimpleAgent())
    # Make the "Free-For-All" environment using the agent list
    return pommerman.make('PommeFFACompetition-v0', agents)

  def reset_tree(self):
    self.tree = {}

  def act(self, obs, action_space):
    action, self.tree = UCT(self, obs, self.tree, action_space, 10)
    return action
