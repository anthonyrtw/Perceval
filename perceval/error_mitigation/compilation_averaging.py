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

from abstract_error_mitigation_technique import ACompilationMitigation


class CompilationAveraging(ACompilationMitigation):
    """
    Circuit Compilation Averaging

    Compile the QPU repeatedly and average output statistics. Exclude
    results based off desired criteria.

    :param reps: Number of extra circuits to compute the average
        output statistics over.
    :param tol: TVD tolerance. Over repetitions, a result, H is
        rejected if TVD(H, H_avg) > μ_TVD + tol * σ_TVD where μ_TVD,
        σ_TVD is the mean & standard dev. of the TVD taken with the
        averaged statistic, H.
    :param leaky_tol: Results with counts detected outside modes of
        input circuit above this threshold are removed.
    """
    def __init__(self, reps: int, tol: float = None, leaky_tol: float = None):
        self.reps = reps
        self.tol = tol
        self.leaky_tol = leaky_tol

