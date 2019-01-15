import argparse
import multiprocessing
import numpy as np
import time

import pommerman
from pommerman.agents import BaseAgent, SimpleAgent, RandomAgent
from pommerman import constants

from pommerman.mcts.mcts_duct import uct as duct
from pommerman.mcts.mcts_exp3 import uct as exp3

NUM_AGENTS = 4

DUCT = 'duct'
EXP3 = 'exp3'

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
        # TODO: make agent act in online game
        assert False

    def play(self):
        self.reset_tree()
        self.env.training_agent = self.agent_id
        obs = self.env.reset()
        state = self.env.get_json_info()
        self.env._init_game_state = state

        length = 0
        done = False
        while not done:
            if args.render:
                self.env.render()

            if args.algorithm == DUCT:
                action, self.tree = duct(self, state, self.tree, args.mcts_iters, args.print_tree)
            else:
                action, self.tree = exp3(self, state, self.tree, args.mcts_iters, args.print_tree)

            if args.render:
                self.env.render()
            actions = self.env.act(obs)
            actions.insert(self.agent_id, action)
            obs, rewards, done, info = self.env.step(actions)
            state = self.env.get_json_info()
            assert self == self.env._agents[self.agent_id]
            length += 1
            print("Agent:", self.agent_id, "Step:", length, "Actions:", [constants.Action(a).name for a in actions], "Rewards:", rewards, "Done:", done)

        reward = rewards[self.agent_id]
        return length, reward, rewards


def runner(id, num_episodes, fifo, _args):
    # make args accessible to MCTSAgent
    global args
    args = _args
    # make sure agents play at all positions
    agent_id = id % NUM_AGENTS
    agent = MctsAgent(agent_id=agent_id)

    for i in range(num_episodes):
        # do rollout
        start_time = time.time()
        length, reward, rewards = agent.play()
        elapsed = time.time() - start_time
        # add data samples to log
        fifo.put((length, reward, rewards, agent_id, elapsed))


def profile_runner(id, num_episodes, fifo, _args):
    import cProfile
    command = """runner(id, num_episodes, fifo, _args)"""
    cProfile.runctx(command, globals(), locals(), filename=_args.profile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile')
    parser.add_argument('--render', action="store_true", default=True)
    # runner params
    parser.add_argument('--num_episodes', type=int, default=1)
    parser.add_argument('--num_runners', type=int, default=1)
    # MCTS params
    parser.add_argument('--mcts_iters', type=int, default=20)
    parser.add_argument('--print_tree', type=bool, default=True)
    parser.add_argument('--mcts_c_puct', type=float, default=1.0)
    parser.add_argument('--algorithm', default='exp3')
    # RL params
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--temperature', type=float, default=0)
    args = parser.parse_args()

    assert args.num_episodes % args.num_runners == 0, "The number of episodes should be divisible by number of runners"

    # use spawn method for starting subprocesses
    ctx = multiprocessing.get_context('spawn')

    # create fifos and processes for all runners
    fifo = ctx.Queue()
    for i in range(args.num_runners):
        process = ctx.Process(target=profile_runner if args.profile else runner, args=(i, args.num_episodes // args.num_runners, fifo, args))
        process.start()

    # do logging in the main process
    all_rewards = []
    all_lengths = []
    all_elapsed = []
    for i in range(args.num_episodes):
        # wait for a new trajectory
        length, reward, rewards, agent_id, elapsed = fifo.get()

        print("Episode:", i, "Reward:", reward, "Length:", length, "Rewards:", rewards, "Agent:", agent_id, "Time per step:", elapsed / length)
        all_rewards.append(reward)
        all_lengths.append(length)
        all_elapsed.append(elapsed)

    print("Average reward:", np.mean(all_rewards))
    print("Average length:", np.mean(all_lengths))
    print("Time per timestep:", np.sum(all_elapsed) / np.sum(all_lengths))

