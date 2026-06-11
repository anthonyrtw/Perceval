"""
Suite for performing arithmetic manipulations on Fock state distributions
and convolutions.
"""
from typing import Sequence

from ....utils.states import BSDistribution, BSCount

Distribution = BSDistribution | BSCount | dict


def sum_distributions(dist_list: list[Distribution], weights: Sequence):
    ...


def convolve_distributions(dist_list: list[Distribution]):
    ...
