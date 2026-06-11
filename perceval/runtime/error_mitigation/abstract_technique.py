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
from abc import ABC, abstractmethod
from statistics import mean

from ...algorithm.parameter_iterator import ParameterIterator
from ...utils.noise_model import NoiseModel


class EMTechnique(ABC):
    """
    Abstract base class for error mitigation techniques.
    """

    @abstractmethod
    def _preprocess(self, iterator: ParameterIterator) -> ParameterIterator:
        """Preprocess the iterator before error mitigation.
        """
        ...

    @abstractmethod
    def _postprocess(self, results_list: list[dict]) -> list[dict]:
        """Postprocess the results of error mitigated jobs.
        """
        ...

    @staticmethod
    def _get_distributions(results_list):
        """Get all the distributions from a results list class.
        """
        return [res["results"] for res in results_list]

    @staticmethod
    def _merge_results(results_list):
        """Average across every entry in a results list.
        Results & iterations are set to None.
        CompiledCircuits and input states are compiled into a set.
        """
        physical_perfs = []
        logical_perfs = []
        global_perfs = []
        transmittances = []
        homs = []
        g2s = []
        compilation = set()
        for res in results_list:
            assert res.get("global_perf") or (res.get("physical_perf") and res.get("logical_perf")), (
                "Result is missing global_perf and/or (physical_perf, logical_perf)"
            )
            # Local simulations return singular global perf
            if "global_perf" in res:
                global_perfs.append(res.get('global_perf'))

            # Remote simulations return logical, physical perf pair.
            elif "logical_perf" in res:
                logical_perfs.append(res.get("logical_perf"))
                physical_perfs.append(res.get("physical_perf"))

            # Noise may not be returned in local simulations.
            if "noise" in res:
                transmittances.append(res["noise"].transmittance)
                homs.append(res["noise"].indistinguishability)
                g2s.append(res["noise"].g2)

            # Compiled Circuit/input only for remote jobs.
            if "compilation" in res:
                compilation.add(res["compilation"])

        # Iteration key appears for parameter iterator jobs/EM jobs.
        # Leave blank for user's input iteration
        out_results = {
            "results": None,
            "iteration": None
        }

        if physical_perfs:
            out_results["physical_perf"] = mean(physical_perfs)
            out_results["logical_perf"] = mean(logical_perfs)
        elif global_perfs:
            out_results["global_perf"] = mean(global_perfs)

        if transmittances:
            out_results["noise"] = NoiseModel(
                indistinguishability=mean(homs),
                transmittance=mean(transmittances),
                g2=mean(g2s)
            )

        elif compilation:
            out_results["compilation"] = compilation

        return out_results


class ChipEMTechnique(EMTechnique):
    """
    Abstract base class for chip error mitigation techniques
    """
    pass
