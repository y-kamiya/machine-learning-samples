import argparse
import torch

NUM_ATOMS_DEFAULT = 1

class Config:
    MODEL_TYPE_FC = 0
    MODEL_TYPE_CONV2D = 1
    DATA_PATH_DEFAULT = 'default.dat'

    def __init__(self, argv):
        parser = argparse.ArgumentParser(add_help=True)
        parser.add_argument('-e', '--env', default='BreakoutNoFrameskip-v4', help='environment name on gym')
        parser.add_argument('-p', '--path', help='path to model data file')
        parser.add_argument('--nosave', action='store_true', help='model parameters are saved')
        parser.add_argument('--steps', type=int, default=108000, help='step count')
        parser.add_argument('--episodes', type=int, default=100, help='episode count')
        parser.add_argument('--epochs', type=int, default=1, help='epoch count')
        parser.add_argument('--atoms', type=int, default=NUM_ATOMS_DEFAULT, help='atom count')
        parser.add_argument('--categorical_v', type=int, default=10, help='Vmin and Vmax of categorical dqn')
        parser.add_argument('--steps_to_update_target', type=int, default=32000, help='count to update target model')
        parser.add_argument('--steps_learning_start', type=int, default=1000, help='start to learn after elapsed this step')
        parser.add_argument('--render', action='store_true', help='render game')
        parser.add_argument('--use_per', action='store_true', help='use prioritized experience replay')
        parser.add_argument('--use_IS', action='store_true', help='use importance sampling with annealing')
        parser.add_argument('--use_noisy_network', action='store_true', help='use noisy network')
        parser.add_argument('--num_multi_step_reward', type=int, default=1, help='multi step reward count')
        parser.add_argument('--cpu', action='store_true', help='use cpu')
        parser.add_argument('--learning_rate', type=float, default=0.00025, help='learning rate for Adam')
        parser.add_argument('--replay_memory_capacity', type=int, default=200000, help='capacity of replay memory')
        parser.add_argument('--batch_size', type=int, default=32, help='size of mini batch')
        parser.add_argument('--gamma', type=float, default=0.99, help='discount rate for reward')
        parser.add_argument('--adam_epsilon', type=float, help='size of mini batch')
        parser.add_argument('--replay_interval', type=int, default=4, help='replay every interval')
        parser.add_argument('--epsilon_start', type=float, default=1.0, help='start value for action exploration')
        parser.add_argument('--epsilon_end', type=float, default=0.01, help='end value for action exploration')
        parser.add_argument('--epsilon_end_step', type=int, default=250000, help='steps to reach epsilon_end')
        args = parser.parse_args(argv)

        self.device_name = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
        self.device = torch.device(self.device_name)

        self.env = args.env
        data_path = args.path
        self.data_path = data_path if data_path != None else self.DATA_PATH_DEFAULT

        self.is_saved = not args.nosave
        self.is_render = args.render
        if self.is_render:
            self.is_saved = False

        self.use_per = args.use_per
        self.use_IS = args.use_IS if self.use_per else False
        self.use_noisy_network = args.use_noisy_network
        self.use_categorical = args.atoms != NUM_ATOMS_DEFAULT
        self.num_multi_step_reward = args.num_multi_step_reward
        self.num_steps = args.steps
        self.num_episodes = args.episodes
        self.num_epochs = args.epochs
        self.num_atoms = args.atoms
        self.categorical_v = args.categorical_v
        self.num_steps_to_update_target = args.steps_to_update_target
        self.steps_learning_start = args.steps_learning_start
        self.replay_memory_capacity = args.replay_memory_capacity
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.adam_epsilon = args.adam_epsilon if args.adam_epsilon != None else 0.01/self.batch_size
        self.replay_interval = args.replay_interval
        self.epsilon_start = args.epsilon_start
        self.epsilon_end = args.epsilon_end
        self.epsilon_end_step = args.epsilon_end_step

        self.model_type = self.MODEL_TYPE_FC

