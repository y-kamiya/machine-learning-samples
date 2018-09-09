import argparse
import sys
import retro

def main(argv=sys.argv[1:]):
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('path', help='path to .bk2 file')
    args = parser.parse_args(argv)

    movie = retro.Movie(args.path)
    movie.step()

    env = retro.make(game=movie.get_game(), use_restricted_actions=retro.ACTIONS_ALL)
    env.initial_state = movie.get_state()
    env.reset()

    while movie.step():
        keys = []
        for i in range(env.NUM_BUTTONS):
            keys.append(movie.get_key(i))
        env.render()
        _obs, _rew, _done, _info = env.step(keys)

if __name__ == '__main__':
    main()

