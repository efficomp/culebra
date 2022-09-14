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

"""NSGA-based wrapper method."""

from __future__ import annotations
from typing import Any, Type, Callable, Tuple, List, Dict, Optional
from deap.algorithms import varAnd
from deap.tools import HallOfFame
from culebra.base import (
    Individual,
    Species,
    FitnessFunction,
    check_int
)
from . import Evolutionary


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2021, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.1.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'

DEFAULT_ELITE_SIZE = 5
"""Default number of elite individuals."""


class Elitist(Evolutionary):
    """Elitist evolutionary algorithm."""

    def __init__(
        self,
        individual_cls: Type[Individual],
        species: Species,
        fitness_function: FitnessFunction,
        num_gens: Optional[int] = None,
        pop_size: Optional[int] = None,
        crossover_func: Optional[
            Callable[[Individual, Individual], Tuple[Individual, Individual]]
        ] = None,
        mutation_func: Optional[
            Callable[[Individual, Individual], Tuple[Individual]]
        ] = None,
        selection_func: Optional[
            Callable[[List[Individual], int, Any], List[Individual]]
        ] = None,
        crossover_prob: Optional[float] = None,
        mutation_prob: Optional[float] = None,
        gene_ind_mutation_prob: Optional[float] = None,
        selection_func_params: Optional[Dict[str, Any]] = None,
        elite_size: Optional[int] = None,
        checkpoint_enable: Optional[bool] = None,
        checkpoint_freq: Optional[int] = None,
        checkpoint_filename: Optional[str] = None,
        verbose: Optional[bool] = None,
        random_seed: Optional[int] = None
    ) -> None:
        """Create a new wrapper.

        :param individual_cls: The individual class
        :type individual_cls: An :py:class:`~base.Individual` subclass
        :param species: The species for all the individuals
        :type species: :py:class:`~base.Species`
        :param fitness_function: The training fitness function
        :type fitness_function: :py:class:`~base.FitnessFunction`
        :param num_gens: The number of generations. If set to
            :py:data:`None`, :py:attr:`~wrapper.DEFAULT_NUM_GENS` will
            be used. Defaults to :py:data:`None`
        :type num_gens: :py:class:`int`, optional
        :param pop_size: The populaion size. If set to
            :py:data:`None`, :py:attr:`~wrapper.single_pop.DEFAULT_POP_SIZE`
            will be used. Defaults to :py:data:`None`
        :type pop_size: :py:class:`int`, greater than zero, optional
        :param crossover_func: The crossover function. If set to
            :py:data:`None`, the *individual_cls*
            :py:meth:`~base.Individual.crossover` method will be used.
            Defaults to :py:data:`None`
        :type crossover_func: :py:class:`~collections.abc.Callable`, optional
        :param mutation_func: The mutation function. If set to
            :py:data:`None`, the *individual_cls*
            :py:meth:`~base.Individual.mutate` method will be used. Defaults
            to :py:data:`None`
        :type mutation_func: :py:class:`~collections.abc.Callable`, optional
        :param selection_func: The selection function. If set to
            :py:data:`None`,
            :py:attr:`~wrapper.single_pop.DEFAULT_SELECTION_FUNC` will be
            used. Defaults to :py:data:`None`
        :type selection_func: :py:class:`~collections.abc.Callable`, optional
        :param crossover_prob: The crossover probability. If set to
            :py:data:`None`,
            :py:attr:`~wrapper.single_pop.DEFAULT_CROSSOVER_PROB` will be
            used. Defaults to :py:data:`None`
        :type crossover_prob: :py:class:`float` in (0, 1), optional
        :param mutation_prob: The mutation probability. If set to
            :py:data:`None`,
            :py:attr:`~wrapper.single_pop.DEFAULT_MUTATION_PROB` will be used.
            Defaults to :py:data:`None`
        :type mutation_prob: :py:class:`float` in (0, 1), optional
        :param gene_ind_mutation_prob: The gene independent mutation
            probability. If set to :py:data:`None`,
            :py:attr:`~wrapper.single_pop.DEFAULT_GENE_IND_MUTATION_PROB` will
            be used. Defaults to :py:data:`None`
        :type gene_ind_mutation_prob: :py:class:`float` in (0, 1), optional
        :param selection_func_params: The parameters for the selection
            function. If set to :py:data:`None`,
            :py:attr:`~wrapper.single_pop.DEFAULT_SELECTION_FUNC_PARAMS` will
            be used. Defaults to :py:data:`None`
        :type selection_func_params: :py:class:`dict`, optional
        :param elite_size: Number of individuals that will be preserved
            as the elite. If set to :py:data:`None`,
            :py:attr:`~wrapper.single_pop.DEFAULT_ELITE_SIZE` will be used.
            Defaults to :py:data:`None`
        :type elite_size: :py:class:`int`, optional
        :param checkpoint_enable: Enable/disable checkpoining. If set to
            :py:data:`None`, :py:attr:`~base.DEFAULT_CHECKPOINT_ENABLE` will
            be used. Defaults to :py:data:`None`
        :type checkpoint_enable: :py:class:`bool`, optional
        :param checkpoint_freq: The checkpoint frequency. If set to
            :py:data:`None`, :py:attr:`~base.DEFAULT_CHECKPOINT_FREQ` will
            be used. Defaults to :py:data:`None`
        :type checkpoint_freq: :py:class:`int`, optional
        :param checkpoint_filename: The checkpoint file path. If set to
            :py:data:`None`, :py:attr:`~base.DEFAULT_CHECKPOINT_FILENAME` will
            be used. Defaults to :py:data:`None`
        :type checkpoint_filename: :py:class:`str`, optional
        :param verbose: The verbosity. If set to
            :py:data:`None`, :py:data:`__debug__` will be used. Defaults to
            :py:data:`None`
        :type verbose: :py:class:`bool`, optional
        :param random_seed: The seed, defaults to :py:data:`None`
        :type random_seed: :py:class:`int`, optional
        :raises TypeError: If any argument is not of the appropriate type
        :raises ValueError: If any argument has an incorrect value
        """
        super().__init__(
            individual_cls,
            species=species,
            fitness_function=fitness_function,
            num_gens=num_gens,
            pop_size=pop_size,
            crossover_func=crossover_func,
            mutation_func=mutation_func,
            selection_func=selection_func,
            crossover_prob=crossover_prob,
            mutation_prob=mutation_prob,
            gene_ind_mutation_prob=gene_ind_mutation_prob,
            selection_func_params=selection_func_params,
            checkpoint_enable=checkpoint_enable,
            checkpoint_freq=checkpoint_freq,
            checkpoint_filename=checkpoint_filename,
            verbose=verbose,
            random_seed=random_seed
        )

        self.elite_size = elite_size

    @property
    def elite_size(self) -> int:
        """Get and set the elite size.

        :getter: Return the elite size
        :setter: Set a new elite size. If set to :py:data:`None`,
            :py:attr:`~wrapper.single_pop.DEFAULT_ELITE_SIZE` is chosen
        :type: :py:class:`int`
        :raises TypeError: If set to a value which is not an int
        :raises ValueError: If set to a value lower than or equal to 0
        """
        return (
            DEFAULT_ELITE_SIZE if self._elite_size is None
            else self._elite_size
        )

    @elite_size.setter
    def elite_size(self, value: int | None) -> None:
        """Set a new value for the elite size.

        :param value: New size. If set to :py:data:`None`,
            :py:attr:`~wrapper.single_pop.DEFAULT_ELITE_SIZE` is chosen
        :type value: :py:class:`int`
        :raises TypeError: If set to a value which is not an int
        :raises ValueError: If *value* is lower than or equal to 0
        """
        # Check the value
        self._elite_size = (
            None if value is None else check_int(value, "elite size", gt=0)
        )

        # Reset the algorithm
        self.reset()

    @property
    def _state(self) -> Dict[str, Any]:
        """Get and set the state of this wrapper.

        Overriden to add the current elite to the wrapper's state.

        :getter: Return the state
        :setter: Set a new state
        :type: :py:class:`dict`
        """
        # Get the state of the superclass
        state = Evolutionary._state.fget(self)

        # Get the state of this class
        state["elite"] = self._elite

        return state

    @_state.setter
    def _state(self, state: Dict[str, Any]) -> None:
        """Set the state of this wrapper.

        Overriden to add the current elite to the wrapper's state.

        :param state: The last loaded state
        :type state: :py:class:`dict`
        """
        # Set the state of the superclass
        Evolutionary._state.fset(self, state)

        # Set the state of this class
        self._elite = state["elite"]

    def _new_state(self) -> None:
        """Generate a new wrapper state.

        Overriden to initialize the elite.
        """
        super()._new_state()

        # Create the elite
        self._elite = HallOfFame(maxsize=min(self.pop_size, self.elite_size))

        # Update the elite
        self._elite.update(self.pop)

    def _reset_state(self) -> None:
        """Reset the wrapper state.

        Overriden to reset the elite.
        """
        super()._reset_state()
        self._elite = None

    def best_solutions(self) -> List[HallOfFame]:
        """Get the best individuals found for each species.

        :return: A list containing :py:class:`~deap.tools.HallOfFame` of
            individuals. One hof for each species
        :rtype: :py:class:`list` of :py:class:`~deap.tools.HallOfFame`
        """
        hof = HallOfFame(maxsize=1)
        if self._elite is not None:
            hof.update(self._elite)
        return [hof]

    def _do_generation(self) -> None:
        """Implement a generation of the search process.

        In this case, the best
        :py:attr:`~wrapper.single_pop.Elitist.elite_size` individuals of each
        generation (the elite) are preserved for the next generation. The
        breeding and selection are implemented as in the
        :py:class:`~wrapper.single_pop.Evolutionary` wrapper.
        """
        # Generate the offspring
        offspring = varAnd(
            self.pop, self._toolbox, self.crossover_prob, self.mutation_prob)

        # Evaluate the individuals with an invalid fitness
        self._evaluate_pop(offspring)
        self._do_generation_stats(offspring)

        # Update the elite
        self._elite.update(offspring)

        # Select the next generation population from offspring and elite
        self.pop[:] = self._toolbox.select(
            offspring + [ind for ind in self._elite],
            self.pop_size
        )


# Exported symbols for this module
__all__ = [
    'Elitist',
    'DEFAULT_ELITE_SIZE'
]
