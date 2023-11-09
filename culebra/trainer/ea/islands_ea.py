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

from typing import (
    Any,
    Type,
    Optional,
    Callable,
    Tuple,
    List,
    Dict,
    Sequence
)

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
        solution_cls: Type[Individual],
        species: Species,
        fitness_function: FitnessFunction,
        subtrainer_cls: Type[SinglePopEA],
        max_num_iters: Optional[int] = None,
        custom_termination_func: Optional[
            Callable[
                [SinglePopEA],
                bool]
        ] = None,
        pop_size: Optional[int] = None,
        crossover_func: Optional[
            Callable[[Individual, Individual], Tuple[Individual, Individual]]
        ] = None,
        mutation_func: Optional[
            Callable[[Individual, float], Tuple[Individual]]
        ] = None,
        selection_func: Optional[
            Callable[[List[Individual], int, Any], List[Individual]]
        ] = None,
        crossover_prob: Optional[float] = None,
        mutation_prob: Optional[float] = None,
        gene_ind_mutation_prob: Optional[float] = None,
        selection_func_params: Optional[Dict[str, Any]] = None,
        num_subtrainers: Optional[int] = None,
        representation_size: Optional[int] = None,
        representation_freq: Optional[int] = None,
        representation_topology_func: Optional[
            Callable[[int, int, Any], List[int]]
        ] = None,
        representation_topology_func_params: Optional[
            Dict[str, Any]
        ] = None,
        representation_selection_func: Optional[
            Callable[[List[Individual], Any], Individual]
        ] = None,
        representation_selection_func_params: Optional[
            Dict[str, Any]
        ] = None,
        checkpoint_enable: Optional[bool] = None,
        checkpoint_freq: Optional[int] = None,
        checkpoint_filename: Optional[str] = None,
        verbose: Optional[bool] = None,
        random_seed: Optional[int] = None,
        **subtrainer_params: Any
    ) -> None:
        """."""
        HomogeneousIslandsEA.__init__(
            self,
            solution_cls,
            species,
            fitness_function,
            subtrainer_cls,
            pop_size=pop_size,
            crossover_func=crossover_func,
            mutation_func=mutation_func,
            selection_func=selection_func,
            crossover_prob=crossover_prob,
            mutation_prob=mutation_prob,
            gene_ind_mutation_prob=gene_ind_mutation_prob,
            selection_func_params=selection_func_params,
            representation_topology_func=representation_topology_func,
            representation_topology_func_params=(
                representation_topology_func_params
            )
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
            representation_selection_func=representation_selection_func,
            representation_selection_func_params=(
                representation_selection_func_params
            ),
            checkpoint_enable=checkpoint_enable,
            checkpoint_freq=checkpoint_freq,
            checkpoint_filename=checkpoint_filename,
            verbose=verbose,
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
        solution_cls: Type[Individual],
        species: Species,
        fitness_function: FitnessFunction,
        subtrainer_cls: Type[SinglePopEA],
        max_num_iters: Optional[int] = None,
        custom_termination_func: Optional[
            Callable[
                [SinglePopEA],
                bool]
        ] = None,
        pop_size: Optional[int] = None,
        crossover_func: Optional[
            Callable[[Individual, Individual], Tuple[Individual, Individual]]
        ] = None,
        mutation_func: Optional[
            Callable[[Individual, float], Tuple[Individual]]
        ] = None,
        selection_func: Optional[
            Callable[[List[Individual], int, Any], List[Individual]]
        ] = None,
        crossover_prob: Optional[float] = None,
        mutation_prob: Optional[float] = None,
        gene_ind_mutation_prob: Optional[float] = None,
        selection_func_params: Optional[Dict[str, Any]] = None,
        num_subtrainers: Optional[int] = None,
        representation_size: Optional[int] = None,
        representation_freq: Optional[int] = None,
        representation_topology_func: Optional[
            Callable[[int, int, Any], List[int]]
        ] = None,
        representation_topology_func_params: Optional[
            Dict[str, Any]
        ] = None,
        representation_selection_func: Optional[
            Callable[[List[Individual], Any], Individual]
        ] = None,
        representation_selection_func_params: Optional[
            Dict[str, Any]
        ] = None,
        checkpoint_enable: Optional[bool] = None,
        checkpoint_freq: Optional[int] = None,
        checkpoint_filename: Optional[str] = None,
        verbose: Optional[bool] = None,
        random_seed: Optional[int] = None,
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
            pop_size=pop_size,
            crossover_func=crossover_func,
            mutation_func=mutation_func,
            selection_func=selection_func,
            crossover_prob=crossover_prob,
            mutation_prob=mutation_prob,
            gene_ind_mutation_prob=gene_ind_mutation_prob,
            selection_func_params=selection_func_params,
            representation_topology_func=representation_topology_func,
            representation_topology_func_params=(
                representation_topology_func_params
            )
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
            representation_selection_func=representation_selection_func,
            representation_selection_func_params=(
                representation_selection_func_params
            ),
            checkpoint_enable=checkpoint_enable,
            checkpoint_freq=checkpoint_freq,
            checkpoint_filename=checkpoint_filename,
            verbose=verbose,
            random_seed=random_seed,
            **subtrainer_params
        )

    # Change the docstring of the constructor to indicate that the default
    # number of subpopulations is the number of CPU cores for parallel
    # multi-population approaches
    __init__.__doc__ = HomogeneousIslandsEA.__init__.__doc__.replace(
        ':py:attr:`~culebra.trainer.DEFAULT_NUM_SUBTRAINERS`',
        'the number of CPU cores'
    )


class HeterogeneousSequentialIslandsEA(
    HeterogeneousIslandsEA,
    SequentialDistributedTrainer
):
    """Sequential island-based model with heterogeneous islands."""

    def __init__(
        self,
        solution_cls: Type[Individual],
        species: Species,
        fitness_function: FitnessFunction,
        subtrainer_cls: Type[SinglePopEA],
        max_num_iters: Optional[int] = None,
        custom_termination_func: Optional[
            Callable[
                [SinglePopEA],
                bool]
        ] = None,
        pop_sizes: Optional[int | Sequence[int]] = None,
        crossover_funcs: Optional[
            Callable[
                [Individual, Individual],
                Tuple[Individual, Individual]
            ] |
            Sequence[
                Callable[
                    [Individual, Individual],
                    Tuple[Individual, Individual]
                ]
            ]
        ] = None,
        mutation_funcs: Optional[
            Callable[
                [Individual, float],
                Tuple[Individual]
            ] |
            Sequence[
                Callable[
                    [Individual, float],
                    Tuple[Individual]
                ]
            ]
        ] = None,
        selection_funcs: Optional[
            Callable[
                [List[Individual], int, Any],
                List[Individual]
            ] | Sequence[
                Callable[
                    [List[Individual], int, Any],
                    List[Individual]
                ]
            ]
        ] = None,
        crossover_probs: Optional[float | Sequence[float]] = None,
        mutation_probs: Optional[float | Sequence[float]] = None,
        gene_ind_mutation_probs: Optional[float | Sequence[float]] = None,
        selection_funcs_params: Optional[
            Dict[str, Any] | Sequence[Dict[str, Any]]
        ] = None,
        num_subtrainers: Optional[int] = None,
        representation_size: Optional[int] = None,
        representation_freq: Optional[int] = None,
        representation_topology_func: Optional[
            Callable[[int, int, Any], List[int]]
        ] = None,
        representation_topology_func_params: Optional[
            Dict[str, Any]
        ] = None,
        representation_selection_func: Optional[
            Callable[[List[Individual], Any], Individual]
        ] = None,
        representation_selection_func_params: Optional[
            Dict[str, Any]
        ] = None,
        checkpoint_enable: Optional[bool] = None,
        checkpoint_freq: Optional[int] = None,
        checkpoint_filename: Optional[str] = None,
        verbose: Optional[bool] = None,
        random_seed: Optional[int] = None,
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
            selection_funcs_params=selection_funcs_params,
            representation_topology_func=representation_topology_func,
            representation_topology_func_params=(
                representation_topology_func_params
            )
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
            representation_selection_func=representation_selection_func,
            representation_selection_func_params=(
                representation_selection_func_params
            ),
            checkpoint_enable=checkpoint_enable,
            checkpoint_freq=checkpoint_freq,
            checkpoint_filename=checkpoint_filename,
            verbose=verbose,
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
        solution_cls: Type[Individual],
        species: Species,
        fitness_function: FitnessFunction,
        subtrainer_cls: Type[SinglePopEA],
        max_num_iters: Optional[int] = None,
        custom_termination_func: Optional[
            Callable[
                [SinglePopEA],
                bool]
        ] = None,
        pop_sizes: Optional[int | Sequence[int]] = None,
        crossover_funcs: Optional[
            Callable[
                [Individual, Individual],
                Tuple[Individual, Individual]
            ] |
            Sequence[
                Callable[
                    [Individual, Individual],
                    Tuple[Individual, Individual]
                ]
            ]
        ] = None,
        mutation_funcs: Optional[
            Callable[
                [Individual, float],
                Tuple[Individual]
            ] |
            Sequence[
                Callable[
                    [Individual, float],
                    Tuple[Individual]
                ]
            ]
        ] = None,
        selection_funcs: Optional[
            Callable[
                [List[Individual], int, Any],
                List[Individual]
            ] | Sequence[
                Callable[
                    [List[Individual], int, Any],
                    List[Individual]
                ]
            ]
        ] = None,
        crossover_probs: Optional[float | Sequence[float]] = None,
        mutation_probs: Optional[float | Sequence[float]] = None,
        gene_ind_mutation_probs: Optional[float | Sequence[float]] = None,
        selection_funcs_params: Optional[
            Dict[str, Any] | Sequence[Dict[str, Any]]
        ] = None,
        num_subtrainers: Optional[int] = None,
        representation_size: Optional[int] = None,
        representation_freq: Optional[int] = None,
        representation_topology_func: Optional[
            Callable[[int, int, Any], List[int]]
        ] = None,
        representation_topology_func_params: Optional[
            Dict[str, Any]
        ] = None,
        representation_selection_func: Optional[
            Callable[[List[Individual], Any], Individual]
        ] = None,
        representation_selection_func_params: Optional[
            Dict[str, Any]
        ] = None,
        checkpoint_enable: Optional[bool] = None,
        checkpoint_freq: Optional[int] = None,
        checkpoint_filename: Optional[str] = None,
        verbose: Optional[bool] = None,
        random_seed: Optional[int] = None,
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
            selection_funcs_params=selection_funcs_params,
            representation_topology_func=representation_topology_func,
            representation_topology_func_params=(
                representation_topology_func_params
            )
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
            representation_selection_func=representation_selection_func,
            representation_selection_func_params=(
                representation_selection_func_params
            ),
            checkpoint_enable=checkpoint_enable,
            checkpoint_freq=checkpoint_freq,
            checkpoint_filename=checkpoint_filename,
            verbose=verbose,
            random_seed=random_seed,
            **subtrainer_params
        )

    # Change the docstring of the constructor to indicate that the default
    # number of subpopulations is the number of CPU cores for parallel
    # multi-population approaches
    __init__.__doc__ = HeterogeneousIslandsEA.__init__.__doc__.replace(
        ':py:attr:`~culebra.trainer.DEFAULT_NUM_SUBTRAINERS`',
        'the number of CPU cores'
    )


__all__ = [
    'HomogeneousSequentialIslandsEA',
    'HomogeneousParallelIslandsEA',
    'HeterogeneousSequentialIslandsEA',
    'HeterogeneousParallelIslandsEA'
]
