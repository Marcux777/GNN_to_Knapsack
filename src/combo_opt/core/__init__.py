"""Core abstractions for combinatorial optimization with GNNs."""

from combo_opt.core.base_decoder import AbstractDecoder, DecodingResult
from combo_opt.core.base_model import AbstractGNNModel
from combo_opt.core.base_problem import OptimizationProblem, ProblemInstance
from combo_opt.core.protocols import Evaluable, Trainable
from combo_opt.core.registry import DecoderRegistry, ModelRegistry

__all__ = [
    "AbstractGNNModel",
    "AbstractDecoder",
    "DecodingResult",
    "OptimizationProblem",
    "ProblemInstance",
    "Trainable",
    "Evaluable",
    "ModelRegistry",
    "DecoderRegistry",
]
