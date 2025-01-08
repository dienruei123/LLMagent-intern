import argparse

dataset_choices = [
    'animals',
    'cities',
    'companies',
    'elements',
    'facts',
    'generated',
    'inventions',
    'all',
    'boolq'
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', choices=range(0, 32), help='The number of layer', type=int, default=19)
    parser.add_argument('--dataset', choices=dataset_choices, help='Dataset name', type=str, default='generated')
    parser.add_argument('--dest', choices=['train', 'test'], help='Destination directory', type=str, default='test')
    parser.add_argument('--model_name', help='Your model name', type=str, default='model')
    parser.add_argument('--ref', action='store_true', help="Add BoolQ Reference text")
    parser.add_argument('--equal', action='store_true', help='Internal State of LLM equal label')
    parser.add_argument('--lamb', choices=[0.005, 0.02, 0.1, 0.2, 0.3, 0.5], help='DA lambda value', type=float, default=0.02)

    args = parser.parse_args()
    return args