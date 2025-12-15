# This file is part of culebra.
#
# Culebra is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# Culebra is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# Culebra. If not, see <http://www.gnu.org/licenses/>.
#
# This work is supported by projects PGC2018-098813-B-C31 and
# PID2022-137461NB-C31, both funded by the Spanish "Ministerio de Ciencia,
# Innovación y Universidades" and by the European Regional Development Fund
# (ERDF).

"""Island-based evolutionary algorithms.

Implementation of several island-based approaches.
"""

from __future__ import annotations

from typing import Any
from collections.abc import Sequence, Callable

from culebra.abc import Species, FitnessFunction
from culebra.solution.abc import Individual
from culebra.trainer.abc import (
    SequentialDistributedTrainer,
    ParallelDistributedTrainer
)
from culebra.trainer.ea.abc import (
    HomogeneousIslandsEA,
    HeterogeneousIslandsEA,
    SinglePopEA
)


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2023, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.3.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


class HomogeneousSequentialIslandsEA(
    HomogeneousIslandsEA,
    SequentialDistributedTrainer
):
    """Sequential island-based model with homogeneous islands."""

    def __init__(
        self,
        solution_cls: type[Individual],
        species: Species,
        fitness_function: FitnessFunction,
        subtrainer_cls: type[SinglePopEA],
        max_num_iters: int | None = None,
        custom_termination_func:
            Callable[[HomogeneousSequentialIslandsEA], bool] | None = None,
        pop_size: int | None = None,
        crossover_func:
            Callable[[Individual, Individual], tuple[Individual, Individual]] |
            None = None,
        mutation_func:
            Callable[[Individual, float], tuple[Individual]] | None = None,
        selection_func:
            Callable[[list[Individual], int, Any], list[Individual]] |
            None = None,
        crossover_prob: float | None = None,
        mutation_prob: float | None = None,
        gene_ind_mutation_prob: float | None = None,
        selection_func_params: dict[str, Any] | None = None,
        num_subtrainers: int | None = None,
        representation_size: int | None = None,
        representation_freq: int | None = None,
        representation_topology_func:
            Callable[[int, int, Any], list[int]] | None = None,
        representation_topology_func_params: dict[str, Any] | None = None,
        representation_selection_func:
            Callable[[list[Individual], Any], Individual] | None = None,
        representation_selection_func_params: dict[str, Any] | None = None,
        checkpoint_activation: bool | None = None,
        checkpoint_freq: int | None = None,
        checkpoint_filename: str | None = None,
        verbosity: bool | None = None,
        random_seed: int | None = None,
        **subtrainer_params: Any
    ) -> None:
        """."""
        HomogeneousIslandsEA.__init__(
            self,
            solution_cls,
            species,
            fitness_function,
            subtrainer_cls,
            num_subtrainers=num_subtrainers,
            pop_size=pop_size,
            crossover_func=crossover_func,
            mutation_func=mutation_func,
            selection_func=selection_func,
            crossover_prob=crossover_prob,
            mutation_prob=mutation_prob,
            gene_ind_mutation_prob=gene_ind_mutation_prob,
            selection_func_params=selection_func_params
        )

        SequentialDistributedTrainer.__init__(
            self,
            fitness_function,
            subtrainer_cls,
            max_num_iters=max_num_iters,
            custom_termination_func=custom_termination_func,
            num_subtrainers=num_subtrainers,
            representation_size=representation_size,
            representation_freq=representation_freq,
            representation_topology_func=representation_topology_func,
            representation_topology_func_params=(
                representation_topology_func_params
            ),
            representation_selection_func=representation_selection_func,
            representation_selection_func_params=(
                representation_selection_func_params
            ),
            checkpoint_activation=checkpoint_activation,
            checkpoint_freq=checkpoint_freq,
            checkpoint_filename=checkpoint_filename,
            verbosity=verbosity,
            random_seed=random_seed,
            **subtrainer_params
        )

    # __init__ shares the documentation with HomogeneousIslandsEA.__init__
    __init__.__doc__ = HomogeneousIslandsEA.__init__.__doc__


class HomogeneousParallelIslandsEA(
        HomogeneousIslandsEA,
        ParallelDistributedTrainer
):
    """Parallel island-based model with homogeneous islands."""

    def __init__(
        self,
        solution_cls: type[Individual],
        species: Species,
        fitness_function: FitnessFunction,
        subtrainer_cls: type[SinglePopEA],
        max_num_iters: int | None = None,
        custom_termination_func:
            Callable[[HomogeneousParallelIslandsEA], bool] | None = None,
        pop_size: int | None = None,
        crossover_func:
            Callable[[Individual, Individual], tuple[Individual, Individual]] |
            None = None,
        mutation_func:
            Callable[[Individual, float], tuple[Individual]] | None = None,
        selection_func:
            Callable[[list[Individual], int, Any], list[Individual]] |
            None = None,
        crossover_prob: float | None = None,
        mutation_prob: float | None = None,
        gene_ind_mutation_prob: float | None = None,
        selection_func_params: dict[str, Any] | None = None,
        num_subtrainers: int | None = None,
        representation_size: int | None = None,
        representation_freq: int | None = None,
        representation_topology_func:
            Callable[[int, int, Any], list[int]] | None = None,
        representation_topology_func_params: dict[str, Any] | None = None,
        representation_selection_func:
            Callable[[list[Individual], Any], Individual] | None = None,
        representation_selection_func_params: dict[str, Any] | None = None,
        checkpoint_activation: bool | None = None,
        checkpoint_freq: int | None = None,
        checkpoint_filename: str | None = None,
        verbosity: bool | None = None,
        random_seed: int | None = None,
        **subtrainer_params: Any
    ) -> None:
        """."""
        # Call both constructors
        HomogeneousIslandsEA.__init__(
            self,
            solution_cls,
            species,
            fitness_function,
            subtrainer_cls,
            num_subtrainers=num_subtrainers,
            pop_size=pop_size,
            crossover_func=crossover_func,
            mutation_func=mutation_func,
            selection_func=selection_func,
            crossover_prob=crossover_prob,
            mutation_prob=mutation_prob,
            gene_ind_mutation_prob=gene_ind_mutation_prob,
            selection_func_params=selection_func_params
        )

        ParallelDistributedTrainer.__init__(
            self,
            fitness_function,
            subtrainer_cls,
            max_num_iters=max_num_iters,
            custom_termination_func=custom_termination_func,
            num_subtrainers=num_subtrainers,
            representation_size=representation_size,
            representation_freq=representation_freq,
            representation_topology_func=representation_topology_func,
            representation_topology_func_params=(
                representation_topology_func_params
            ),
            representation_selection_func=representation_selection_func,
            representation_selection_func_params=(
                representation_selection_func_params
            ),
            checkpoint_activation=checkpoint_activation,
            checkpoint_freq=checkpoint_freq,
            checkpoint_filename=checkpoint_filename,
            verbosity=verbosity,
            random_seed=random_seed,
            **subtrainer_params
        )


class HeterogeneousSequentialIslandsEA(
    HeterogeneousIslandsEA,
    SequentialDistributedTrainer
):
    """Sequential island-based model with heterogeneous islands."""

    def __init__(
        self,
        solution_cls: type[Individual],
        species: Species,
        fitness_function: FitnessFunction,
        subtrainer_cls: type[SinglePopEA],
        max_num_iters: int | None = None,
        custom_termination_func:
            Callable[[HeterogeneousSequentialIslandsEA], bool] | None = None,
        pop_sizes: int | Sequence[int] | None = None,
        crossover_funcs:
            Callable[
                [Individual, Individual], tuple[Individual, Individual]
            ] |
            Sequence[
                Callable[
                    [Individual, Individual], tuple[Individual, Individual]
                ]
            ] |
            None = None,
        mutation_funcs:
            Callable[[Individual, float], tuple[Individual]] |
            Sequence[Callable[[Individual, float], tuple[Individual]]] |
            None = None,
        selection_funcs:
            Callable[[list[Individual], int, Any], list[Individual]] |
            Sequence[
                Callable[[list[Individual], int, Any], list[Individual]]
            ] |
            None = None,
        crossover_probs: float | Sequence[float] | None = None,
        mutation_probs: float | Sequence[float] | None = None,
        gene_ind_mutation_probs: float | Sequence[float] | None = None,
        selection_funcs_params:
            dict[str, Any] | Sequence[dict[str, Any]] | None = None,
        num_subtrainers: int | None = None,
        representation_size: int | None = None,
        representation_freq: int | None = None,
        representation_topology_func:
            Callable[[int, int, Any], list[int]] | None = None,
        representation_topology_func_params: dict[str, Any] | None = None,
        representation_selection_func:
            Callable[[list[Individual], Any], Individual] | None = None,
        representation_selection_func_params: dict[str, Any] | None = None,
        checkpoint_activation: bool | None = None,
        checkpoint_freq: int | None = None,
        checkpoint_filename: str | None = None,
        verbosity: bool | None = None,
        random_seed: int | None = None,
        **subtrainer_params: Any
    ) -> None:
        """."""
        # Call both constructors
        HeterogeneousIslandsEA.__init__(
            self,
            solution_cls,
            species,
            fitness_function,
            subtrainer_cls,
            num_subtrainers=num_subtrainers,
            pop_sizes=pop_sizes,
            crossover_funcs=crossover_funcs,
            mutation_funcs=mutation_funcs,
            selection_funcs=selection_funcs,
            crossover_probs=crossover_probs,
            mutation_probs=mutation_probs,
            gene_ind_mutation_probs=gene_ind_mutation_probs,
            selection_funcs_params=selection_funcs_params
        )

        SequentialDistributedTrainer.__init__(
            self,
            fitness_function,
            subtrainer_cls,
            max_num_iters=max_num_iters,
            custom_termination_func=custom_termination_func,
            num_subtrainers=num_subtrainers,
            representation_size=representation_size,
            representation_freq=representation_freq,
            representation_topology_func=representation_topology_func,
            representation_topology_func_params=(
                representation_topology_func_params
            ),
            representation_selection_func=representation_selection_func,
            representation_selection_func_params=(
                representation_selection_func_params
            ),
            checkpoint_activation=checkpoint_activation,
            checkpoint_freq=checkpoint_freq,
            checkpoint_filename=checkpoint_filename,
            verbosity=verbosity,
            random_seed=random_seed,
            **subtrainer_params
        )

    # __init__ shares the documentation with HeterogeneousIslandsEA.__init__
    __init__.__doc__ = HeterogeneousIslandsEA.__init__.__doc__


class HeterogeneousParallelIslandsEA(
    HeterogeneousIslandsEA,
    ParallelDistributedTrainer
):
    """Parallel island-based model with heterogeneous islands."""

    def __init__(
        self,
        solution_cls: type[Individual],
        species: Species,
        fitness_function: FitnessFunction,
        subtrainer_cls: type[SinglePopEA],
        max_num_iters: int | None = None,
        custom_termination_func:
            Callable[[HeterogeneousParallelIslandsEA], bool] | None = None,
        pop_sizes: int | Sequence[int] | None = None,
        crossover_funcs:
            Callable[
                [Individual, Individual], tuple[Individual, Individual]
            ] |
            Sequence[
                Callable[
                    [Individual, Individual], tuple[Individual, Individual]
                ]
            ] |
            None = None,
        mutation_funcs:
            Callable[[Individual, float], tuple[Individual]] |
            Sequence[Callable[[Individual, float], tuple[Individual]]] |
            None = None,
        selection_funcs:
            Callable[[list[Individual], int, Any], list[Individual]] |
            Sequence[
                Callable[[list[Individual], int, Any], list[Individual]]
            ] |
            None = None,
        crossover_probs: float | Sequence[float] | None = None,
        mutation_probs: float | Sequence[float] | None = None,
        gene_ind_mutation_probs: float | Sequence[float] | None = None,
        selection_funcs_params:
            dict[str, Any] | Sequence[dict[str, Any]] | None = None,
        num_subtrainers: int | None = None,
        representation_size: int | None = None,
        representation_freq: int | None = None,
        representation_topology_func:
            Callable[[int, int, Any], list[int]] | None = None,
        representation_topology_func_params: dict[str, Any] | None = None,
        representation_selection_func:
            Callable[[list[Individual], Any], Individual] | None = None,
        representation_selection_func_params: dict[str, Any] | None = None,
        checkpoint_activation: bool | None = None,
        checkpoint_freq: int | None = None,
        checkpoint_filename: str | None = None,
        verbosity: bool | None = None,
        random_seed: int | None = None,
        **subtrainer_params: Any
    ) -> None:
        """."""
        # Call both constructors
        HeterogeneousIslandsEA.__init__(
            self,
            solution_cls,
            species,
            fitness_function,
            subtrainer_cls,
            num_subtrainers=num_subtrainers,
            pop_sizes=pop_sizes,
            crossover_funcs=crossover_funcs,
            mutation_funcs=mutation_funcs,
            selection_funcs=selection_funcs,
            crossover_probs=crossover_probs,
            mutation_probs=mutation_probs,
            gene_ind_mutation_probs=gene_ind_mutation_probs,
            selection_funcs_params=selection_funcs_params
        )

        ParallelDistributedTrainer.__init__(
            self,
            fitness_function,
            subtrainer_cls,
            max_num_iters=max_num_iters,
            custom_termination_func=custom_termination_func,
            num_subtrainers=num_subtrainers,
            representation_size=representation_size,
            representation_freq=representation_freq,
            representation_topology_func=representation_topology_func,
            representation_topology_func_params=(
                representation_topology_func_params
            ),
            representation_selection_func=representation_selection_func,
            representation_selection_func_params=(
                representation_selection_func_params
            ),
            checkpoint_activation=checkpoint_activation,
            checkpoint_freq=checkpoint_freq,
            checkpoint_filename=checkpoint_filename,
            verbosity=verbosity,
            random_seed=random_seed,
            **subtrainer_params
        )


__all__ = [
    'HomogeneousSequentialIslandsEA',
    'HomogeneousParallelIslandsEA',
    'HeterogeneousSequentialIslandsEA',
    'HeterogeneousParallelIslandsEA'
]
