"""Central registries for public pipeline axes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


class RegistryError(ValueError):
    """Raised when a requested registry entry is not supported."""


@dataclass(frozen=True)
class RegistryEntry(Generic[T]):
    name: str
    description: str
    factory: Callable[[], T]


class Registry(Generic[T]):
    def __init__(self, axis_name: str) -> None:
        self.axis_name = axis_name
        self._entries: dict[str, RegistryEntry[T]] = {}

    def register(self, name: str, description: str, factory: Callable[[], T]) -> None:
        if name in self._entries:
            raise RegistryError(f"duplicate {self.axis_name} registry entry: {name}")
        self._entries[name] = RegistryEntry(name=name, description=description, factory=factory)

    def get(self, name: str) -> T:
        try:
            return self._entries[name].factory()
        except KeyError as exc:
            supported = ", ".join(sorted(self._entries)) or "(none)"
            raise RegistryError(
                f"unsupported {self.axis_name}: {name}. Supported values: {supported}"
            ) from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))


benchmark_registry: Registry[Any] = Registry("benchmark")
collection_registry: Registry[Any] = Registry("collection method")
dataset_registry: Registry[Any] = Registry("dataset method")
training_registry: Registry[Any] = Registry("training method")
evaluation_registry: Registry[Any] = Registry("evaluation planner")


def register_defaults() -> None:
    """Populate global registries once."""

    if benchmark_registry.names():
        return

    from .evaluation.benchmarks.babaisai import BabaisaiAdapter
    from .evaluation.benchmarks.babyai import BabyAIAdapter
    from .evaluation.benchmarks.webshop import WebShopAdapter
    from .evaluation.collection.dtc import DTCCollector
    from .evaluation.collection.none import NoOpCollector
    from .data.aliases import (
        DivPOFreqPairBuilder,
        DivPOProbPairBuilder,
        TieDPODavPairBuilder,
        TieDPORKPairBuilder,
    )
    from .data.dpo_pairs import DPOPairBuilder
    from .data.divpo_pairs import DivPOPairBuilder
    from .data.noop_pairs import NoOpPairBuilder
    from .data.rto_pairs import RTOPairBuilder
    from .data.tiedpo_pairs import TieDPOPairBuilder
    from .evaluation.suites import EvaluationPlanner
    from .training.aliases import (
        BaseModelTrainer,
        DDOTrainer,
        DivPOFreqTrainer,
        DivPOProbTrainer,
        ReferenceSFTTrainer,
        TieDPODavTrainer,
        TieDPORKTrainer,
    )
    from .training.dpo import DPOTrainer
    from .training.divpo import DivPOTrainer
    from .training.rto import RTOTrainer
    from .training.tiedpo import TieDPOTrainer

    benchmark_registry.register("babyai", "BabyAI benchmark adapter", BabyAIAdapter)
    benchmark_registry.register("babaisai", "Baba Is AI benchmark adapter", BabaisaiAdapter)
    benchmark_registry.register("babaisyou", "Alias for the Baba Is AI benchmark adapter", BabaisaiAdapter)
    benchmark_registry.register("webshop", "WebShop benchmark adapter", WebShopAdapter)

    collection_registry.register("none", "No collection", NoOpCollector)
    collection_registry.register("dtc", "Divergence Tree Collection", DTCCollector)
    dataset_registry.register("noop_pairs", "No-op dataset for base/reference rows", NoOpPairBuilder)
    dataset_registry.register("dpo_pairs", "DPO win-lose pair dataset builder", DPOPairBuilder)
    dataset_registry.register("divpo_pairs", "DivPO pair dataset builder", DivPOPairBuilder)
    dataset_registry.register("divpo_freq_pairs", "DivPO-freq pair dataset builder", DivPOFreqPairBuilder)
    dataset_registry.register("divpo_prob_pairs", "DivPO-prob pair dataset builder", DivPOProbPairBuilder)
    dataset_registry.register("rto_pairs", "RTO pair target dataset builder", RTOPairBuilder)
    dataset_registry.register("tiedpo_pairs", "TieDPO pair dataset builder", TieDPOPairBuilder)
    dataset_registry.register("tiedpo_rk_pairs", "TieDPO-RK pair dataset builder", TieDPORKPairBuilder)
    dataset_registry.register("tiedpo_dav_pairs", "TieDPO-Dav pair dataset builder", TieDPODavPairBuilder)
    training_registry.register("base", "Base model checkpoint manifest", BaseModelTrainer)
    training_registry.register("reference", "SFT reference checkpoint manifest", ReferenceSFTTrainer)
    training_registry.register("dpo", "DPO LoRA trainer", DPOTrainer)
    training_registry.register("divpo", "DivPO LoRA trainer", DivPOTrainer)
    training_registry.register("divpo_freq", "DivPO-freq LoRA trainer", DivPOFreqTrainer)
    training_registry.register("divpo_prob", "DivPO-prob LoRA trainer", DivPOProbTrainer)
    training_registry.register("rto", "Reference-relative target odds trainer", RTOTrainer)
    training_registry.register("ddo", "DDO reference-relative target trainer", DDOTrainer)
    training_registry.register("tiedpo", "TieDPO LoRA trainer", TieDPOTrainer)
    training_registry.register("tiedpo_rk", "TieDPO-RK LoRA trainer", TieDPORKTrainer)
    training_registry.register("tiedpo_dav", "TieDPO-Dav LoRA trainer", TieDPODavTrainer)
    evaluation_registry.register("default", "Evaluation", EvaluationPlanner)
