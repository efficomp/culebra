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
# This work was supported by project PGC2018-098813-B-C31 (Spanish "Ministerio
# de Ciencia, Innovación y Universidades"), and by the European Regional
# Development Fund (ERDF).

"""Cooperative co-evolutionary algorithm.

Implementation of a distributed cooperative co-evolutionary algorithm.
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

from culebra.abc import (
    Species,
    FitnessFunction
)
from culebra.solution.abc import Individual
from culebra.trainer.abc import (
    SequentialMultiPopTrainer,
    ParallelMultiPopTrainer
)
from culebra.trainer.ea.abc import SinglePopEA, CooperativeEA


__author__ = "Jesús González"
__copyright__ = "Copyright 2023, EFFICOMP"
__license__ = "GNU GPL-3.0-or-later"
__version__ = "0.2.1"
__maintainer__ = "Jesús González"
__email__ = "jesusgonzalez@ugr.es"
__status__ = "Development"


class SequentialCooperativeEA(CooperativeEA, SequentialMultiPopTrainer):
    """Sequential implementation of the cooperative evolutionary trainer."""

    def __init__(
        self,
        solution_classes: Type[Individual] | Sequence[Type[Individual]],
        species: Species | Sequence[Species],
        fitness_function: FitnessFunction,
        subpop_trainer_cls: Type[SinglePopEA],
        max_num_iters: Optional[int] = None,
        custom_termination_func: Optional[
            Callable[
                [SinglePopEA],
                bool
            ]
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
        num_subpops: Optional[int] = None,
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
        **subpop_trainer_params: Any
    ) -> None:
        """."""
        CooperativeEA.__init__(
            self,
            solution_classes=solution_classes,
            species=species,
            fitness_function=fitness_function,
            subpop_trainer_cls=subpop_trainer_cls,
            num_subpops=num_subpops,
            pop_sizes=pop_sizes,
            crossover_funcs=crossover_funcs,
            mutation_funcs=mutation_funcs,
            selection_funcs=selection_funcs,
            crossover_probs=crossover_probs,
            mutation_probs=mutation_probs,
            gene_ind_mutation_probs=gene_ind_mutation_probs,
            selection_funcs_params=selection_funcs_params,
        )

        SequentialMultiPopTrainer.__init__(
            self,
            fitness_function,
            subpop_trainer_cls,
            max_num_iters=max_num_iters,
            custom_termination_func=custom_termination_func,
            num_subpops=num_subpops,
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
            checkpoint_enable=checkpoint_enable,
            checkpoint_freq=checkpoint_freq,
            checkpoint_filename=checkpoint_filename,
            verbose=verbose,
            random_seed=random_seed,
            **subpop_trainer_params
        )

    # __init__ shares the documentation with CooperativeEA.__init__
    __init__.__doc__ = CooperativeEA.__init__.__doc__


class ParallelCooperativeEA(CooperativeEA, ParallelMultiPopTrainer):
    """Parallel implementation of the cooperative evolutionary trainer."""

    def __init__(
        self,
        solution_classes: Type[Individual] | Sequence[Type[Individual]],
        species: Species | Sequence[Species],
        fitness_function: FitnessFunction,
        subpop_trainer_cls: Type[SinglePopEA],
        max_num_iters: Optional[int] = None,
        custom_termination_func: Optional[
            Callable[
                [SinglePopEA],
                bool
            ]
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
        num_subpops: Optional[int] = None,
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
        **subpop_trainer_params: Any
    ) -> None:
        """."""
        CooperativeEA.__init__(
            self,
            solution_classes=solution_classes,
            species=species,
            fitness_function=fitness_function,
            subpop_trainer_cls=subpop_trainer_cls,
            num_subpops=num_subpops,
            pop_sizes=pop_sizes,
            crossover_funcs=crossover_funcs,
            mutation_funcs=mutation_funcs,
            selection_funcs=selection_funcs,
            crossover_probs=crossover_probs,
            mutation_probs=mutation_probs,
            gene_ind_mutation_probs=gene_ind_mutation_probs,
            selection_funcs_params=selection_funcs_params,
        )

        ParallelMultiPopTrainer.__init__(
            self,
            fitness_function,
            subpop_trainer_cls,
            max_num_iters=max_num_iters,
            custom_termination_func=custom_termination_func,
            num_subpops=num_subpops,
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
            checkpoint_enable=checkpoint_enable,
            checkpoint_freq=checkpoint_freq,
            checkpoint_filename=checkpoint_filename,
            verbose=verbose,
            random_seed=random_seed,
            **subpop_trainer_params
        )

    # __init__ shares the documentation with CooperativeEA.__init__
    __init__.__doc__ = CooperativeEA.__init__.__doc__


__all__ = [
    'SequentialCooperativeEA',
    'ParallelCooperativeEA'
]