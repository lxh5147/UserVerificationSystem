#test arg parse
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument(
    '--filters',
    type=int,
    nargs='+',
    default=[64, 128, 256, 512],
    help='filters')
parser.add_argument(
        '--blocks',
        type=int,
        default=3,
        help='blocks')
parser.add_argument(
        '--like',
        type=bool,
        default=False,
        help='like')

FLAGS=parser.parse_args(['--blocks=5', '--filters','3','5','--like=True'])
print(FLAGS.blocks)
print(FLAGS.filters)
print(FLAGS.like)