import argparse
import torch

NUM_EPISODES_DEFAULT = 100
NUM_STEPS_DEFAULT = 3000
NUM_EPOCHS_DEFAULT = 1
NUM_STEPS_TO_UPDATE_TARGET_DEFAULT = 1
DATA_PATH_DEFAULT = 'default.dat'

class Config:
    def __init__(self, argv):
        parser = argparse.ArgumentParser(add_help=True)
        parser.add_argument('-p', '--path', help='path to model data file')
        parser.add_argument('--nosave', action='store_true', help='model parameters are saved')
        parser.add_argument('--steps', type=int, help='step count')
        parser.add_argument('--episodes', type=int, help='episode count')
        parser.add_argument('--epochs', type=int, help='epoch count')
        parser.add_argument('--steps_to_update_target', type=int, help='count to update target model')
        parser.add_argument('--render', action='store_true', help='render game')
        parser.add_argument('--use_per', action='store_true', help='render game')
        parser.add_argument('--cpu', action='store_true', help='use cpu')
        args = parser.parse_args(argv)

        device_name = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
        self.device = torch.device(device_name)

        data_path = args.path
        self.data_path = data_path if data_path != None else DATA_PATH_DEFAULT

        self.is_saved = not args.nosave
        self.is_render = args.render
        self.use_per = args.use_per
        self.num_steps = args.steps if args.steps != None else NUM_STEPS_DEFAULT
        self.num_episodes = args.episodes if args.episodes != None else NUM_EPISODES_DEFAULT
        self.num_epochs = args.epochs if args.epochs != None else NUM_EPOCHS_DEFAULT
        self.num_steps_to_update_target = args.steps_to_update_target if args.steps_to_update_target != None else NUM_STEPS_TO_UPDATE_TARGET_DEFAULT

