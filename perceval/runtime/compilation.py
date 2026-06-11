from typing import Callable

from exqalibur import FockState
from ..components.compiled_circuit import CompiledCircuit
from ..components.experiment import Experiment
from ..utils.states import BSCount, BSDistribution

Distribution = BSDistribution | BSCount | dict


class Compilation:
    """
    Experiment compilation

    Compilation contains compiled circuit info, how input states are mapped to
    input modes, how output modes of a circuit are mapped to output modes of
    PIC, how compiled states are decoded, and how distributions are
    postprocessed with respect to Postselection, heralds, min detected photons.
    """
    def __init__(
        self,
        compiled_circuit: CompiledCircuit,
        compiled_input: FockState,
        input_modes: list[int],
        output_modes: list[int],
    ):
        self._compiled_circuit = compiled_circuit
        self._compiled_input = compiled_input
        self._input_modes = input_modes
        self._output_modes = output_modes

    def decompile_results(self, raw_results: Distribution) -> Distribution:
        """Postprocess a given set of results with respect to both the output
        mapping function and the provided post-selection postprocessing function.
        """
        for k in raw_results.keys():
            assert k.m == self._compiled_circuit.m, ("Raw results contains a "
                f"key whose length does not match compiled circuit {k.m} != {self._compiled_circuit.m}"
            )

        # First map output modes to correct logical outputs
        decoded = type(raw_results)({
            self.output_mapping_fn(state): v
            for state, v in raw_results.items()
        })
        return decoded

    def input_mapping_fn(self, state: FockState) -> FockState:
        assert state.m == self.experiment.m, (
            "User input state does not match experiment shape."
        )
        return self._slice_fock_state(state, self._input_modes)

    def output_mapping_fn(self, state: FockState) -> FockState:
        assert state.m == self.compiled_circuit.m, (
            "Raw output state does not match compiled circuit shape."
        )
        return self._slice_fock_state(state, self._output_modes)

    @property
    def compiled_circuit(self) -> CompiledCircuit:
        return self._compiled_circuit

    @property
    def compiled_input(self) -> FockState:
        return self._compiled_input

    def _slice_fock_state(self, fock_state: FockState, indices) -> FockState:
        return FockState([fock_state[i] for i in indices])
