import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='p2d', choices=['p2d', 'd2p'])
    parser.add_argument('--flag', action='store_true')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--hotspot', action='store_true')
    group.add_argument('--grid', action='store_true')
    args = parser.parse_args()

    print(args.flag)