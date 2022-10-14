"""
Merges csv from different methods into a single csv file
(like it was in results in initial dqn_zoo repo)

Usage sample:
python explore/merge_csv.py --environments breakout pong --base addqn_50mln --directory logs
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


def merge_csv(base_dir, base_name, environments):
    df = pd.DataFrame({
        attribute: [] for attribute in ATTRIBUTES
    })
    for environment_name in environments:
        method_df = pd.read_csv(base_dir / f"{base_name}_{environment_name}.csv")
        method_df['environment_name'] = environment_name
        method_df = method_df[ATTRIBUTES]
        df = pd.concat((df, method_df))
        print(len(df))

    return df

def main():
    args = parse_args()
    base_dir = Path(args.directory)
    df = merge_csv(base_dir, args.base, args.environments)
    df.to_csv(base_dir / f'{args.base}.csv')


if __name__ == '__main__':
    main()


