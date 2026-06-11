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
from typing import Self

from .abstract_technique import ErrorMitigationTechnique
from .photon_error_mitigation import PhotonErrorMitigation
from .compilation_averaging import CompilationAveraging
from .preselection import Preselection
from .detector_balancing import DetectorBalancing
from .loss_mitigation import LossMitigation
from .utils._error_mitigation_utils import experiment_to_iterator
from ...algorithm.parameter_iterator import ParameterIterator
from ...components.experiment import Experiment



class ErrorMitigation:
    """
    Error mitigation settings.

    Stores a number of error mitigation techniques which are used to prepare
    the experiment and post-process the results. Techniques are applied in a
    fixed order: preprocessing runs in reverse postprocessing order, while raw
    results are postprocessed from detector-level corrections toward
    compilation-level averaging or preselection.
    """

    POSTPROCESSING_ORDER = [
        "DetectorBalancing",
        "LossMitigation",
        "PhotonRecycling",
        "PhotonErrorMitigation",
        "CompilationAveraging",
        "Preselection"
    ]

    def __init__(self):
        self._settings = {}

    def add(self, *techniques: ErrorMitigationTechnique) -> Self:
        """Add error mitigation techniques to the settings."""
        for t in techniques:
            self._validate_technique(t)
            self._settings[t.__class__.__name__] = t
        return self

    def add_photon_error_mitigation(self, order: int) -> Self:
        technique = PhotonErrorMitigation(order)
        return self.add(technique)

    def add_compilation_averaging(self, reps: int, tol: float) -> Self:
        technique = CompilationAveraging(reps, tol)
        return self.add(technique)

    def add_detector_balancing(self) -> Self:
        technique = DetectorBalancing()
        return self.add(technique)

    def add_loss_mitigation(self) -> Self:
        technique = LossMitigation()
        return self.add(technique)
    
    @property
    def techniques(self):
        return list(self._settings.values())

    def samples_budget(
        self,
        max_samples: int,
        max_shots: int,
        experiment: Experiment = None,
        iterator: ParameterIterator = None
    ) -> tuple[list[int], list[int]]:
        """Return how samples and shots are split over error-
        mitigation subjobs.

        :param max_samples: Number of requested samples.
        :param max_shots: Shot limit per job.
        :param experiment: Input error-mitigated linear-optical experiment
        :param iterator: Parameter iterator that encodes a sequence of
            experiments.
        """
        if experiment is not None and iterator.experiment != experiment:
            raise ValueError("experiment and iterator.experiment do not match.")

        iterator = (
            iterator.iterations if iterator
            else experiment_to_iterator(experiment, max_samples, max_shots)
        )
        for cls in reversed(self.POSTPROCESSING_ORDER):
            technique = self._settings.get(cls)
            if technique is not None:
                iterator = technique._update_iterator(iterator, context={})

        budget = [
            {
                "input_state": it["input_state"],
                "max_samples": it["max_samples"],
                "max_shots": it["max_shots"],
                "min_detected_photons": it["min_detected_photons"]
            }
            for it in iterator.iterations
        ]
        return budget

    def _postprocess(self, results_list: list[dict], context) -> list[dict]:
        """Perform post-processing on raw results to mitigate errors."""
        for cls in self.POSTPROCESSING_ORDER:
            technique = self._settings.get(cls)
            if technique is not None:
                results_list = technique._postprocess(results_list, context)

        return results_list

    def _preprocess(self, iterator: ParameterIterator, context) -> ParameterIterator:
        for cls in reversed(self.POSTPROCESSING_ORDER):
            technique = self._settings.get(cls)
            if technique is not None:
                iterator, context = technique._preprocess(iterator, context)

        return iterator, context

    def _validate_technique(self, t: ErrorMitigationTechnique):
        settings = self._settings
        if (
            isinstance(t, LossMitigation) and "PhotonErrorMitigation" in settings or
            isinstance(t, PhotonErrorMitigation) and "LossMitigation" in settings
        ):
            raise ValueError(
                "Loss mitigation & photon error mitigation cannot be enabled "
                "simultaneously.")
        if (
            isinstance(t, CompilationAveraging) and "Preselection" in settings or
            isinstance(t, Preselection) and "CompilationAveraging" in settings
        ):
            raise ValueError(
                "Compilation averaging & preselection cannot be enabled "
                "simultaneously.")
