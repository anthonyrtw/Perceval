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
import copy

from .abstract_technique import ChipEMTechnique
from .utils._distributions import sum_distributions
from .utils._error_mitigation_utils import batch_list
from ...utils.dist_metrics import tvd_dist
from ...utils.logging import get_logger, channel


class CompilationAveraging(ChipEMTechnique):
    """
    Compilation Averaging

    Compile and run the same experiment several times and average the
    resulting statistics. Optional tolerances can reject compilations whose 
    distribution is too far from the batch reference, or distributions which 
    contain leaky above a certain tolerance.

    :param reps: Number of compilations to average per input job.
    :param tol: Maximum total variation distance from the batch average.
    :param leaky_tol: Maximum tolerated leaky probability mass.
    """
    def __init__(self, reps: int, tol: float = None, leaky_tol: float = None):
        assert isinstance(reps, int), "`reps` must be of type `int`."
        assert reps > 0, "`reps` must be greater than 0."
        assert isinstance(tol, (float, None)), "`tol` must be of type: `float` or `None`."
        assert isinstance(leaky_tol, (float, None)), "`leaky_tol` must be of type: `float` or `None`."

        self._reps = reps
        self._tol = tol
        self._leaky_tol = leaky_tol

    def _preprocess(self, iterator, context):
        new_iterator = copy.copy(iterator)
        new_iterator._iterations = []

        for iteration in iterator._iterations:
            old_seed = iteration.get("seed", 0)
            max_samples = iteration.get("max_samples", iterator._max_samples)
            max_shots = iteration.get("max_shots", iterator._max_shots)

            max_samples_rep = self._split_budget(max_samples, self._reps, "max_samples")
            max_shots_rep = self._split_budget(max_shots, self._reps, "max_shots")

            for i, (max_samples, max_shots) in enumerate(zip(max_samples_rep, max_shots_rep)):
                new_iteration = iteration.copy()
                new_iteration["max_samples"] = max_samples
                new_iteration["max_shots"] = max_shots
                new_iteration["seed"] = old_seed + i
                new_iterator.add_iteration(**new_iteration)

        context.error_mitigation["CompilationAveraging"] = {
            "reps": self._reps,
            "tol": self._tol,
            "leaky_tol": self._leaky_tol
        }
        return new_iterator, context

    @classmethod
    def _postprocess(cls, results_list, context):
        reps = context.error_mitigation["CompilationAveraging"]["reps"]
        tol = context.error_mitigation["CompilationAveraging"]["tol"]
        leaky_tol = context.error_mitigation["CompilationAveraging"]["leaky_tol"]

        # Group raw results list into experiment batches
        results_batched = batch_list(results_list, reps)

        new_results_list = []
        for res_batch in results_batched:
            weights = [1 / reps] * reps
            dists_batch = cls._get_distributions(res_batch)

            reference = sum_distributions(dists_batch, weights=weights)

            keep = []
            for dist in dists_batch:

                # Check for outlier or leaky distributions.
                if (
                    (tol is None or tvd_dist(reference, dist) <= tol) and
                    (leaky_tol is None or not cls._is_leaky_dist(dist, leaky_tol, context))
                ):
                    keep.append(dist)

            if not keep:
                get_logger().info(
                    "Cannot reject results based on tol without rejecting "
                    "every result in batch. Keeping all results.", channel
                )
                averaged = reference
            elif len(keep) == len(res_batch):
                averaged = reference
            else:
                weights = [1 / len(keep)] * len(keep)
                averaged = sum_distributions(keep, weights=weights)

            new_results = cls._merge_results(res_batch)
            new_results["results"] = averaged
            new_results_list.append(new_results)

        return new_results_list
    
    @staticmethod
    def _is_leaky_dist(dist, leaky_tol, context):
        norm = sum(dist.values())
        output_modes = context.output_modes
        qpu_size = context.platform.m
        qpu_modes = list(range(qpu_size))

        leaky_sum = 0
        for k, v in dist.items():
            leaky_count = sum([k[i] for i in qpu_modes if i not in output_modes])

            if leaky_count:
                leaky_sum += v

        return leaky_sum / norm >= leaky_tol

    @staticmethod
    def _split_budget(total: int, reps: int, name: str) -> list[int]:
        if total < reps:
            raise ValueError(
                f"{name}={total} is too low to split across {reps} "
                "compilation-averaging jobs."
            )

        # Ensure total number of samples is conserved between splits.
        base, rem = divmod(total, reps)
        return [base + int(i < rem) for i in range(reps)]