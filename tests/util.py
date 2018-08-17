from parlai.core.agents import Agent
import random
from os.path import expanduser
home = expanduser("~")

min_opt = {'download_path': home + '/ParlAI/downloads',
           'image_mode': 'raw',
           'include_labels': True,
           'override': {},
           'parlai_home': home + '/ParlAI',
           'datatype': 'train',
           'batchsize': 1,
           'datapath': home + '/ParlAI/data',
           'numthreads': 1,
           'task': None}


class RandomAgent(Agent):
    def __init__(self, opt):
        self.id = 'RandomAgent'
        self.opt = opt

    def act(self):
        reply = {'id': self.id}
        if 'label_candidates' in self.observation:
            cands = list(self.observation['label_candidates'])
            reply['text'] = random.choice(cands)
        else:
            reply['text'] = "I don't know."
        return reply


class RepeatLabelAgent(Agent):
    def __init__(self, opt):
        self.id = 'RepeatLabelAgent'

    def act(self):
        reply = {'id': self.id}
        if 'labels' in self.observation:
            reply['text'] = ', '.join(self.observation['labels'])
        else:
            reply['text'] = "I don't know."
        return reply


def basic_world_exploration(world,
                            iterations=10,
                            verbose=False):
    """
    loop to calculate the reward funtcion
    """
    acc_reward = 0
    hard_reward = 0
    labels_count = 0
    reward_count = 0
    for _ in range(iterations):
        world.parley()
        obs = world.acts[0]
        action = world.acts[1]
        if "labels" in obs:
            labels_count += 1
            hard_reward += int(obs["labels"][0] == action["text"])
        if "reward" in obs:
            reward_count += 1
            acc_reward += obs["reward"]
        if verbose:
            print("\nenv: {} ".format(obs["text"]))
            print("agent: {} ".format(action["text"]))
            print("acc_reward = {} \n".format(acc_reward))
        if world.epoch_done():
            print('EPOCH DONE')
            break
    r_per_iteration = acc_reward / iterations
    r_per_reward_count = acc_reward / max(1, reward_count)
    hard_r_per_iteration = hard_reward / iterations
    hard_r_per_labels_count = hard_reward / max(1, labels_count)
    return r_per_iteration, r_per_reward_count, hard_r_per_iteration, hard_r_per_labels_count  # noqa