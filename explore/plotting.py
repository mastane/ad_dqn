"""
Plots graphics for different rl methods from csv result files
Usage example
python explore/plotting.py --environment_name 'pong' --num_iterations=50 --smoothing=3 --note="0.8"
"""
"""
Lvl 4:
Put on provided axis
"""
from pathlib import Path
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib

METHODS = ['dqn', 'double_q', 'qrdqn', 'c51', 'caddqn']

NAMES_MAP = {
    'dqn': "DQN",
    'double_q': 'Double DQN',
    'qrdqn': 'QR DQN',
    'addqn': 'SAD DQN',
    'saddqn': 'SAD DQN',
    'maddqn': 'MAD DQN',
    'c51': 'c51',
    'caddqn': 'CAD DQN',
}


def pretty_matplotlib_config(fontsize=15):
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams.update({'font.size': fontsize})


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--num_iterations', type=int, default=50)
    parser.add_argument('--environment_name', type=str, default='alien')
    parser.add_argument('--base_directory', type=str, default='logs')
    parser.add_argument('--smoothing', type=int, default=0)
    parser.add_argument('--note', type=str, default='')
    parser.add_argument('--pdf', default=False, action='store_true')
    args = parser.parse_args()
    return args



def single_plot(axes, methods, base_dir, environment_name, num_iterations, smoothing):
    for method in methods:
        df = pd.read_csv(base_dir / f'{method}.csv')
        df = df[df.environment_name == environment_name]
        df = df[df.frame < (num_iterations+1)*1e6]

        grouped = df.groupby('frame')
        frames = grouped.frame.mean() / 1e6
        mean_score = grouped['eval_episode_return'].mean()
        min_score = grouped['eval_episode_return'].min()
        max_score = grouped['eval_episode_return'].max()

        if smoothing != 0:
            mean_score = mean_score.rolling(smoothing).mean()
            min_score = min_score.rolling(smoothing).mean()
            max_score = max_score.rolling(smoothing).mean()

        axes.plot(frames, mean_score, label=NAMES_MAP[method])
        axes.fill_between(frames, min_score, max_score, alpha=0.2)
    axes.grid()



    axes.set_title(environment_name.capitalize())


def main():
    args = parse_args()
    print(args)
    base_directory = Path(args.base_directory)

    pretty_matplotlib_config(24)
    figure = plt.figure(figsize=(10, 8))
    axes = figure.add_subplot(111)

    single_plot(axes, METHODS, base_directory, args.environment_name, args.num_iterations, args.smoothing)
    plt.xlabel('Million frames')
    plt.ylabel('Evaluation episode return')
    plt.legend()

    if args.pdf:
        plt.savefig(base_directory / 'figures' / f'{args.environment_name}_{args.note}.pdf', dpi=120)
    else:
        plt.savefig(base_directory / 'figures' / f'{args.environment_name}_{args.note}.png', dpi=120)
    plt.show()


if __name__ == '__main__':
    main()


