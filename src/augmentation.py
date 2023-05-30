import argparse

from data_augmentation import paraphrase_and_evaluate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="LTP project",
    )

    subparsers = parser.add_subparsers(help='commands', title="commands", dest="command")

    generation_parser = subparsers.add_parser('generation', help='generation parameters')

    generation_parser.add_argument(
        '--csv_path',
        default='arguments-training.tsv'
        type=str
    )

    generation_parser.add_argument(
        '--gpu',
        action='store_true'
    )

    generation_parser.add_argument(
        '--num_beams',
        default=5,
        type=int,
        help='number of search beams')
    
    generation_parser.add_argument(
        '--num_beam_groups',
        default=5,
        type=int,
        help='number of groups to split beams into'
    )

    generation_parser.add_argument(
        '--num_return_sequences',
        default=5,
        type=int,
        help='number of paraphrases to return'
    )

    generation_parser.add_argument(
        '--repetition_penalty',
        default=10.0,
        type=float
    )

    generation_parser.add_argument(
        '--diversity_penalty',
        default=3.0,
        type=float
    )

    generation_parser.add_argument(
        '--no_repeat_ngram_size',
        default=2,
        type=int
    )

    generation_parser.add_argument(
        '--temperature',
        default=0.7,
        type=float
    )

    generation_parser.add_argument(
        '--max_length',
        default=128,
        type=int
    )

    args = parser.parse_args()

    paraphrase_and_evaluate(args)