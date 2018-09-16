import argparse
import torch

NUM_EPISODES_DEFAULT = 100
NUM_STEPS_DEFAULT = 4500
NUM_EPOCHS_DEFAULT = 1
NUM_STEPS_TO_UPDATE_TARGET_DEFAULT = 1
NUM_MULTI_STEP_REWARD_DEFAULT = 1
LEARNING_RATE_DEFAULT = 0.0001
STEPS_LEARNING_START_DEFAULT = 1000
REPLY_MEMORY_CAPACITY_DEFAULT = 10000

class Config:
    MODEL_TYPE_FC = 0
    MODEL_TYPE_CONV2D = 1
    DATA_PATH_DEFAULT = 'default.dat'

    def __init__(self, argv):
        parser = argparse.ArgumentParser(add_help=True)
        parser.add_argument('-p', '--path', help='path to model data file')
        parser.add_argument('--nosave', action='store_true', help='model parameters are saved')
        parser.add_argument('--steps', type=int, help='step count')
        parser.add_argument('--episodes', type=int, help='episode count')
        parser.add_argument('--epochs', type=int, help='epoch count')
        parser.add_argument('--steps_to_update_target', type=int, help='count to update target model')
        parser.add_argument('--steps_learning_start', type=int, help='start to learn after elapsed this step')
        parser.add_argument('--render', action='store_true', help='render game')
        parser.add_argument('--use_per', action='store_true', help='use prioritized experience reply')
        parser.add_argument('--use_noisy_network', action='store_true', help='use noisy network')
        parser.add_argument('--num_multi_step_reward', type=int, help='multi step reward count')
        parser.add_argument('--cpu', action='store_true', help='use cpu')
        parser.add_argument('--learning_rate', type=float, help='learning rate for Adam')
        parser.add_argument('--reply_memory_capacity', type=int, help='capacity of reply memory')
        args = parser.parse_args(argv)

        self.device_name = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
        self.device = torch.device(self.device_name)

        data_path = args.path
        self.data_path = data_path if data_path != None else self.DATA_PATH_DEFAULT

        self.is_saved = not args.nosave
        self.is_render = args.render
        if self.is_render:
            self.is_saved = False

        self.use_per = args.use_per
        self.use_noisy_network = args.use_noisy_network
        self.num_multi_step_reward = args.num_multi_step_reward if args.num_multi_step_reward != None else NUM_MULTI_STEP_REWARD_DEFAULT
        self.num_steps = args.steps if args.steps != None else NUM_STEPS_DEFAULT
        self.num_episodes = args.episodes if args.episodes != None else NUM_EPISODES_DEFAULT
        self.num_epochs = args.epochs if args.epochs != None else NUM_EPOCHS_DEFAULT
        self.num_steps_to_update_target = args.steps_to_update_target if args.steps_to_update_target != None else NUM_STEPS_TO_UPDATE_TARGET_DEFAULT
        self.steps_learning_start = args.steps_learning_start if args.steps_learning_start != None else STEPS_LEARNING_START_DEFAULT
        self.reply_memory_capacity = args.reply_memory_capacity if args.reply_memory_capacity != None else REPLY_MEMORY_CAPACITY_DEFAULT
        self.learning_rate = args.learning_rate if args.learning_rate != None else LEARNING_RATE_DEFAULT

        self.model_type = self.MODEL_TYPE_FC
