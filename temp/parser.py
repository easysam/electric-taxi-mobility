import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--foo', action='store_true')
    group.add_argument('--bar', action='store_true')
    args = parser.parse_args()
    print(args.foo)
