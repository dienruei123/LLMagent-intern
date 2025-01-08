import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--ref', action='store_true', help="Add BoolQ Reference text")
args = parser.parse_args()

print(args.ref)
