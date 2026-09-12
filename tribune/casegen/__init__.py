"""Privacy-preserving synthetic case generator, conformal verification, & world model."""

from .conformal import (
    ConformalCalibrationResult,
    ConformalCalibrator,
    ConformalScoreType,
)
from .simulation import (
    SimulationEngine,
    SimulationState,
    SimulationTurn,
    TrajectoryOutcome,
    WitnessDepositionSimulator,
)
from .synthetic import (
    DualAgentScenarioMiner,
    ResearchAgent,
    ScenarioAgent,
    SyntheticCaseGenerator,
    SyntheticEnvironment,
)
from .world_model import (
    CourtroomWorldModel,
    StatutoryInvariant,
    StatutoryWorldModel,
)

__all__ = [
    "ConformalCalibrationResult",
    "ConformalCalibrator",
    "ConformalScoreType",
    "SimulationEngine",
    "SimulationState",
    "SimulationTurn",
    "TrajectoryOutcome",
    "WitnessDepositionSimulator",
    "DualAgentScenarioMiner",
    "ResearchAgent",
    "ScenarioAgent",
    "SyntheticCaseGenerator",
    "SyntheticEnvironment",
    "CourtroomWorldModel",
    "StatutoryInvariant",
    "StatutoryWorldModel",
]
