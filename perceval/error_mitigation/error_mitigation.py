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

from typing import Self, Sequence
import warnings

from .abstract_error_mitigation_technique import AErrorMitigationTechnique
from .photon_error_mitigation import PhotonErrorMitigation
from .compilation_averaging import CompilationAveraging
from .compilation_preselection import CompilationPreselection
from .detector_balancing import DetectorBalancing
from .loss_mitigation import PhotonRecycling
from ..utils import BasicState


class ErrorMitigation:
    """
    Error Mitigation configuration.

    Controls the extent and type of error mitigation applied during
    simulation or execution.

    :param int tier:
        Preset level of error mitigation.

        **Tier 0**
            No error mitigation is performed.

        **Tier 1**
            - Photon error mitigation with correction order = 1
            - Compilation error mitigation via averaging with ``reps = 5``
            - Loss balancing enabled

        **Tier 2**
            - Photon error mitigation with correction order = 2
            - Compilation error mitigation via averaging with ``reps = 10``
            - Loss balancing enabled

        **Tier 3**
            - Photon error mitigation with correction order = 2
            - Compilation error mitigation via averaging with ``reps = 10``
            - Loss balancing enabled

    Error mitigation settings can also be manually tuned using the
    fluent configuration methods provided by this class.

    Example::

        error_mitigation = (
            ErrorMitigation(tier=1)
            .mitigate_photon_errors(order=2)
            .compilation_averaging(False)
            .compilation_preselection(max_accept=2, n_trials=5)
            .loss_balancing(False)
        )
    """

    def __init__(self, tier: int):
        if tier not in (0, 1, 2, 3):
            raise ValueError("Error mitigation tier must be in [0, 1, 2, 3].")

        self.tier = tier
        self._instantiate_techniques(tier)

    def add_mitigation(self, *args: AErrorMitigationTechnique) -> Self:
        """Add error mitigation techniques.
        """
        if len({type(arg) for arg in args}) != len(args):
            raise TypeError(
                f"Multiple arguments passed for the same error mitigation"
                "technique.")

        # Ensure that compilation averaging + preselection are not both applied
        apply_avg = any(isinstance(arg, CompilationAveraging) for arg in args)
        apply_pre = any(isinstance(arg, CompilationPreselection) for arg in args)

        if (apply_avg and (apply_pre or self.compilation_preselection)) or (
            apply_pre and self.compilation_preselection
        ):
            raise NotImplementedError(
                "Cannot apply compilation averaging & pre-selection "
                "simultaneously.")

        for obj in args:
            if isinstance(obj, PhotonErrorMitigation):
                self.photon_error_mitigation = obj

            elif isinstance(obj, CompilationPreselection):
                if self.compilation_averaging:
                    warnings.warning(
                        "Cannot use both compilation preselection and averaging. "
                        "Removing averaging.")
                    self.compilation_averaging = None
                self.compilation_preselection = obj

            elif isinstance(obj, CompilationAveraging):
                if self.compilation_preselection:
                    warnings.warning(
                        "Cannot use both compilation preselection and averaging. "
                        "Removing preselection.")
                self.compilation_averaging = obj

            elif isinstance(obj, DetectorBalancing):
                self.loss_balancing = obj

            else:
                raise ValueError(
                    f"Unrecognized Error Mitigation technique of type: {type(obj)}."
                )
        return self

    def reset(self) -> Self:
        self.photon_error_mitigation = None
        self.compilation_averaging = None
        self.compilation_preselection = None
        self.loss_balancing = None
        self.photon_recycling = None
        return self

    def mitigate_photon_errors(self, enable: bool = True, order: int = None) -> Self:
        """
        Partial Distinguishability and g(2) Mitigation

        Uses partition mitigation framework to prepare extra sub-input
        states and postprocess accordingly.

        :param order: Partition mitigation order. If a `list`, the specified
            distinguishability orders are corrected. If an `int`, every
            correction <= order is applied. If `-1`, all `n` orders for
            `n` photons are corrected.
        """
        if enable:
            if order is None:
                raise ValueError("Please a value for order.")

            photon_error_mitigation = PhotonErrorMitigation(order)
            self.add_mitigation(photon_error_mitigation)
        else:
            self.photon_error_mitigation = None
        return self

    def average_compilations(
        self,
        enable: bool = True,
        reps: int = None,
        tol: float = None,
        leaky_tol: float = None
    ) -> Self:
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
        if enable:
            if not reps:
                raise ValueError("Please specify a number of repetitions.")

            compilation_averaging = CompilationAveraging(reps, tol, leaky_tol)
            self.add_mitigation(compilation_averaging)
        else:
            self.compilation_averaging = None
        return self

    def preselect_compilations(
        self,
        enable: bool = True,
        n_samples: int = None,
        n_trials: int = None,
        max_accept: int = None,
        train: bool = True,
        subspace: Sequence[BasicState] = None
    ) -> Self:
        """
        Circuit Compilation Pre-selection

        Pre-compiles a circuit a number of times & accept unitaries based on
        their accuracy in single-photon experiments.

        :param enable: Enable/disable compilation pre-selection.
        :param n_samples: Number of samples to obtain in each pre-selection
            experiment.
        :param n_trials: Number of compilations to test.
        :param max_accept: Max. number of compilations to accept. Fixed for
            `train=False`.
        :param train: Take a weighted average of the results at run-time.
            Weights calculated from single-photon results via training
            at pre-selection phase.
        :param subspace: a target list of BasicState to compute the TVD for
            preselection

        """
        if enable:
            if not n_samples:
                raise ValueError(
                    "Please specify `n_samples` for pre-selection experiments.")

            if not n_trials:
                raise ValueError(
                    "Please specify `n_trials` for pre-selection experiments.")

            if not max_accept:
                raise ValueError(
                    "Please specify `max_accept` for pre-selection experiments.")

            preselection = CompilationPreselection(n_samples, n_trials, max_accept, train, subspace)
            self.add_mitigation(preselection)
        else:
            self.preselection = None

        return self

    def enable_photon_recycling(self, enable: bool) -> Self:
        """
        Loss mitigation via Photon Recycling.
        """

        if enable:
            self.add_mitigation(PhotonRecycling())
        else:
            self.photon_recycling = None
        return self

    def enable_detector_balancing(self, enable: bool) -> Self:
        """
        Detector loss correction.

        Adjust the probabilities of each output state based on the output
        loss of each mode.
        """
        if enable:
            self.add_mitigation(DetectorBalancing())
        else:
            self.loss_balancing = None
        return self

    @property
    def max_overhead(self) -> int:
        """Indicates the maximum possible number of error mitigation
        subjobs for a single error mitigated job.

        The true overhead is dependent on the input state for photon-
        error mitigation and the outcome of the pre-selection process.
        """
        overhead = 1

        # Input state dependent - but take maximum possible based on order.
        if self.photon_error_mitigation:
            overhead *= self.photon_error_mitigation.max_overhead

        elif self.compilation_averaging:
            overhead *= self.compilation_averaging.reps

        # Multiply by the maximum possible number of repeated compilations.
        elif self.compilation_preselection:
            max_accept = self.compilation_preselection.max_accept
            overhead *= max_accept

        return overhead

    @property
    def settings(self):
        return self.__dict__(self)

    def __dict__(self):
        photon_mit = self.photon_error_mitigation and dict(self.photon_error_mitigation)
        comp_avg = self.compilation_averaging and dict(self.compilation_averaging)
        comp_pre = self.compilation_preselection and dict(self.compilation_preselection)

        return {
            "photon_error_mitigation": photon_mit,
            "compilation_averaging": comp_avg,
            "compilation_preselection": comp_pre,
            "loss_balancing": bool(self.loss_balancing),
            "photon_recycling": bool(self.photon_recycling)
        }

    def _instantiate_techniques(self):
        """Enable different error mitigation techniques based on the tier.
        """
        self.reset()

        if self.tier == 1:
            self.add_mitigation(
                PhotonErrorMitigation(order=1),
                CompilationAveraging(reps=5, tol=1e-3, leaky_tol=1e-3),
                DetectorBalancing(),
            )
        elif self.tier == 2:
            self.add_mitigation(
                PhotonErrorMitigation(order=1),
                CompilationPreselection(reps=10, tol=1e-3, max_accept=2),
                DetectorBalancing(),
            )
        elif self.tier == 3:
            self.add_mitigation(
                PhotonErrorMitigation(order="max"),
                CompilationPreselection(reps=10, tol=1e-3, max_accept=2),
                DetectorBalancing()
            )

    def _prepare_batch_job(self, iterator: list[dict]):
        """Add jobs to a batch-job iterator with respect to specified
        error mitigation techniques.
        """
        if self.compilation_averaging:
            iterator = self.compilation_averaging._prepare_iterator(iterator)

        elif self.compilation_preselection:
            iterator = self.compilation_preselection._prepare_iterator(iterator)

        if self.photon_error_mitigation:
            iterator = self.photon_error_mitigation._prepare_iterator(iterator)

        if self.photon_recycling:
            ...

        return iterator

    def _postprocess(self, results_list: list[dict]):
        """Apply error mitigation postprocessing to results list.
        """
        if self.loss_balancing:
            results_list = self.loss_balancing._postprocess(results_list)

        if self.photon_error_mitigation:
            results_list = self.photon_error_mitigation._postprocess(results_list)

        if self.compilation_averaging:
            results_list = self.compilation_averaging._postprocess(results_list)

        elif self.compilation_preselection:
            results_list = self.compilation_preselection._postprocess(results_list)

        return results_list
