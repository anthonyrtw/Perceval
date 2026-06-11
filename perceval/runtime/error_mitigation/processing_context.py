from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from ..compilation import Compilation
from ...components.experiment import Experiment


# Simulation backend, Ascella, Belenos.
Platform = ...


@dataclass
class ProcessingContext(ABC):
    """Store backend information used for mitigation postprocessing."""
    
    @abstractmethod
    @property
    def output_modes(self):
        ...

    @abstractmethod
    @property
    def detection_pattern(self):
        ...



@dataclass
class LocalContext(ProcessingContext):
    """Context for local error-mitigation postprocessing.

    On local computation, the provided circuit is imagined to represent the 
    "architecture" of the chip.
    """
    experiment: Experiment
    error_mitigation: dict[str, dict] | None = field(default_factory=dict)

    @property
    def output_modes(self):
        return list(range(self.experiment.circuit_size))

    @property
    def detection_pattern(self):
        return [
            i if i is None
            else i.max_detections
            for i in self.experiment._detectors
        ]
    
    @property
    def output_losses(self):
        # Wait on support for losses in components.Detector
        ...


@dataclass
class RemoteContext(ProcessingContext):
    """Context for remote error-mitigation postprocessing."""
    platform: Platform
    compilation: Compilation
    error_mitigation: dict[str, dict] | None = field(default_factory=dict)

    @property
    def noise(self):
        return self.platform.get_current_noise()

    @property
    def output_modes(self):
        return self.compilation.output_modes

    @property
    def detection_pattern(self):
        pnr_pattern = [
            detector.max_detections if detector is not None
            else None
            for detector in self.platform.specs.architecture.detectors
        ]
        return pnr_pattern

    @property
    def output_losses(self):
        self.platform.performance["'Transmission per mode (%)'"]
