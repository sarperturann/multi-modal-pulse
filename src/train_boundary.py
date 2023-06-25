"""Trains semantic boundary from latent space.

Basically, this file takes a collection of `latent code - attribute score`
pairs, and find the separation boundary by treating it as a bi-classification
problem and training a linear SVM classifier. The well-trained decision boundary
of the SVM classifier will be saved as the boundary corresponding to a
particular semantic from the latent space. The normal direction of the boundary
can be used to manipulate the correpsonding attribute of the synthesis.
"""

import os.path
import argparse
import numpy as np

from manipulator import train_boundary
from pathlib import Path
out_path = Path("our_boundaries")
out_path.mkdir(parents=True, exist_ok=True)


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        description='Train semantic boundary with given latent codes and '
                    'attribute scores.')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                        help='Directory to save the output results. (required)')
    parser.add_argument('-c', '--latent_codes_path', type=str, required=True,
                        help='Path to the input latent codes. (required)')
    parser.add_argument('-s', '--scores_path', type=str, required=True,
                        help='Path to the input attribute scores. (required)')
    parser.add_argument('-n', '--chosen_num_or_ratio', type=float, default=0.02,
                        help='How many samples to choose for training. '
                             '(default: 0.2)')
    parser.add_argument('-r', '--split_ratio', type=float, default=0.7,
                        help='Ratio with which to split training and validation '
                             'sets. (default: 0.7)')
    parser.add_argument('-V', '--invalid_value', type=float, default=None,
                        help='Sample whose attribute score is equal to this '
                             'field will be ignored. (default: None)')

    parser.add_argument('-bn', '--boundary_name', type=str, default="smile", required=False,
                        help='Boundary type/name must be given with this field. Default is smile')
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    if not os.path.isfile(args.latent_codes_path):
        raise ValueError(
            f'Latent codes `{args.latent_codes_path}` does not exist!')
    latent_codes = np.load(args.latent_codes_path)

    if not os.path.isfile(args.scores_path):
        raise ValueError(
            f'Attribute scores `{args.scores_path}` does not exist!')
    scores = np.load(args.scores_path)

    boundary = train_boundary(latent_codes=latent_codes,
                              scores=scores,
                              chosen_num_or_ratio=args.chosen_num_or_ratio,
                              split_ratio=args.split_ratio,
                              invalid_value=args.invalid_value)
    np.save(os.path.join(args.output_dir,
            f"boundary_{args.boundary_name}.npy"), boundary)


if __name__ == '__main__':
    main()
