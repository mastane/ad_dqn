import os
import sys
from pathlib import Path
from argparse import ArgumentParser

from matplotlib import pyplot as plt

sys.path.append('.')
from explore.merge_csv import merge_csv
from explore.plotting import single_plot, METHODS, pretty_matplotlib_config



def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--environments', nargs='+', default=['breakout', 'robotank', 'pong'],
        help='Space separate list of environments to merge'
    )
    parser.add_argument('--method', type=str, default='addqn')
    parser.add_argument("--note", type=str, default='50mln', help='name modifier in the middle')
    parser.add_argument('--source_directory', type=str, default='results', help='Directory with source csv files')
    parser.add_argument('--result_directory', type=str, default='logs', help='Directory with other method files')

    parser.add_argument('--num_iterations', type=int, default=50)
    parser.add_argument('--environment_name', type=str, default='alien')
    parser.add_argument('--base_directory', type=str, default='logs')
    parser.add_argument('--smoothing', type=int, default=0)
    parser.add_argument('--result_note', type=str, default='')
    parser.add_argument('--pdf', default=False, action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    source_directory = Path(args.source_directory)
    df = merge_csv(source_directory, args.method, args.note, args.environments)
    result_directory = Path(args.result_directory)
    df.to_csv(result_directory / f"{args.method}.csv")
    print(df)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    pretty_matplotlib_config(16)
    fig.tight_layout(rect=(0.05, 0.05, 0.95, 0.93))
    # fig.tight_layout()
    print(axes.shape)
    print(axes)

    for i, env in enumerate(args.environments):
        print(i, env)
        j = i // 3
        k = i  - j * 3
        ax = axes[i//3, i - (i//3)*3]
        single_plot(ax, METHODS, result_directory, env, args.num_iterations, args.smoothing)

    fig.supxlabel('Million frames')
    fig.supylabel('Evaluation episode return')


    # handles, labels = ax.get_legend_handles_label()
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncols=6)

    # plt.xlabel('Million frames')
    # plt.ylabel('Evaluation episode return')
    # plt.legend()
    fig_path = result_directory / 'figures' / args.result_note
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    plt.savefig( fig_path / f'{args.method}.pdf', dpi=120)
    plt.show()


if __name__ == '__main__':
    main()

