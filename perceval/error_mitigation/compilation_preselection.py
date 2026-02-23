# MIT License
#
# Copyright (c) 2022 Quandela
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# As a special exception, the copyright holders of exqalibur library give you
# permission to combine exqalibur with code included in the standard release of
# Perceval under the MIT license (or modified versions of such code). You may
# copy and distribute such a combined system following the terms of the MIT
# license for both exqalibur and Perceval. This exception for the usage of
# exqalibur is limited to the python bindings used by Perceval.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Sequence

from .abstract_error_mitigation_technique import ACompilationMitigation
from ..utils import BasicState


class CompilationPreselection(ACompilationMitigation):
    """
    Circuit Compilation Pre-selection

    Pre-compiles a circuit a number of times & accept unitaries based on
    their accuracy in single-photon experiments.

    :param n_samples: Number of samples to obtain in each pre-selection
        experiment.
    :param n_trials: Number of compilations to test.
    :param max_accept: Max. number of compilations to accept. Number of
        compilations is fixed for `train=False`.
    :param train: Take a weighted average of the results at run-time.
        Weights calculated from single-photon results via training
        at pre-selection phase.
    :param subspace: a target list of BasicState to compute the TVD for
        preselection
    """
    def __init__(
        self,
        n_samples: int,
        n_trials: int,
        max_accept: int,
        train: bool,
        subspace: Sequence[BasicState]
    ):
        self.nsamples = n_samples
        self.ntrials = n_trials
        self.max_accept = max_accept
        self.train = train
        self.subspace = tuple(subspace)
