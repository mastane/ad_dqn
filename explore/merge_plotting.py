from argparse import ArgumentParser
from explore.merge_csv import merge_csv


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



