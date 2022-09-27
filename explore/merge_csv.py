"""
flags array?
"""
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd


ATTRIBUTES = ['frame', 'environment_name', 'eval_episode_return']


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--environments', nargs='+', default=['breakout', 'robotank', 'pong'],
        help='Space separate list of environments to merge'
    )
    parser.add_argument(
        '--base', type=str, default='addqn_50mln',
        help='Starts of name for files, i.e. like in "saddqn_50mln_pong.csv" it would be "saddqn_50mln"'
    )
    parser.add_argument('--directory', type=str, default='logs', help='Directory with files')
    return parser.parse_args()

def main():
    """
    Usage sample:
    python explore/merge_csv.py --environments breakout pong --base addqn_50mln --directory logs
    """
    args = parse_args()
    base_dir = Path(args.directory)
    df = pd.DataFrame({
        attribute: [] for attribute in ATTRIBUTES
    })
    for name in args.environments:
        method_df = pd.read_csv(base_dir / f"{args.base}_{name}.csv")
        method_df['environment_name'] = name
        method_df = method_df[ATTRIBUTES]
        df = pd.concat((df, method_df))
        print(len(df))

    df.to_csv(base_dir / 'addqn.csv')



if __name__ == '__main__':
    main()


