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

"""Island-based evolutionary wrapper.

Implementation of a distributed evolutionary algorithm where the population
is divided into several isolated subpopulations (islands).
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
    Sequence)
from copy import deepcopy
from functools import partial
from itertools import repeat
from deap.tools import HallOfFame
from culebra.base import (
    Individual,
    Species,
    FitnessFunction,
    check_int,
    check_float,
    check_func,
    check_func_params,
    check_sequence
)
from culebra.wrapper.single_pop import SinglePop
from . import MultiPop, SequentialMultiPop, ParallelMultiPop
from .topology import ring_destinations

__author__ = "Jesús González"
__copyright__ = "Copyright 2021, EFFICOMP"
__license__ = "GNU GPL-3.0-or-later"
__version__ = "0.1.1"
__maintainer__ = "Jesús González"
__email__ = "jesusgonzalez@ugr.es"
__status__ = "Development"


DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC = ring_destinations
"""Default topology function for the islands model."""

DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC_PARAMS = {}
"""Parameters for the default topology function in the islands model."""


class Islands(MultiPop):
    """Abstract island-based model for evolutionary wrappers."""

    def __init__(
        self,
        individual_cls: Type[Individual],
        species: Species,
        fitness_function: FitnessFunction,
        subpop_wrapper_cls: Type[SinglePop],
        num_gens: Optional[int] = None,
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
        **subpop_wrapper_params: Any
    ) -> None:
        """Create a new wrapper.

        :param individual_cls: The individual class
        :type individual_cls: An :py:class:`~base.Individual` subclass
        :param species: The species for all the individuals
        :type species: :py:class:`~base.Species`
        :param fitness_function: The training fitness function
        :type fitness_function: :py:class:`~base.FitnessFunction`
        :param subpop_wrapper_cls: Single-population wrapper class to handle
            the subpopulations (islands).
        :type subpop_wrapper_cls: Any subclass of
            :py:class:`~wrapper.single_pop.SinglePop`
        :param num_gens: The number of generations. If set to
            :py:data:`None`, :py:attr:`~wrapper.DEFAULT_NUM_GENS` will
            be used. Defaults to :py:data:`None`
        :type num_gens: :py:class:`int`, optional
        :param num_subpops: The number of subpopulations (islands). If set to
            :py:data:`None`, :py:attr:`~wrapper.multi_pop.DEFAULT_NUM_SUBPOPS`
            will be used. Defaults to :py:data:`None`
        :type num_subpops: :py:class:`int`, optional
        :param representation_size: Number of representative individuals that
            will be sent to the other subpopulations. If set to
            :py:data:`None`,
            :py:attr:`~wrapper.multi_pop.DEFAULT_REPRESENTATION_SIZE` will be
            used. Defaults to :py:data:`None`
        :type representation_size: :py:class:`int`, optional
        :param representation_freq: Number of generations between
            representatives sendings. If set to :py:data:`None`,
            :py:attr:`~wrapper.multi_pop.DEFAULT_REPRESENTATION_FREQ` will be
            used. Defaults to :py:data:`None`
        :type representation_freq: :py:class:`int`, optional
        :param representation_freq: Number of generations between
            representatives sendings. If set to :py:data:`None`,
            :py:attr:`~wrapper.multi_pop.DEFAULT_REPRESENTATION_FREQ` will be
            used. Defaults to :py:data:`None`
        :type representation_freq: :py:class:`int`, optional
        :param representation_topology_func: Topology function for
            representatives sending. If set to :py:data:`None`,
            :py:attr:`~wrapper.multi_pop.DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC`
            will be used. Defaults to :py:data:`None`
        :type representation_topology_func:
            :py:class:`~collections.abc.Callable`, optional
        :param representation_topology_func_params: Parameters to obtain the
            destinations with the topology function. If set to :py:data:`None`,
            :py:attr:`~wrapper.multi_pop.DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC_PARAMS`
            will be used. Defaults to :py:data:`None`
        :type representation_topology_func_params: :py:class:`dict`, optional
        :param representation_selection_func: Policy function to choose the
            representatives from each subpopulation (island). If set to
            :py:data:`None`,
            :py:attr:`~wrapper.multi_pop.DEFAULT_REPRESENTATION_SELECTION_FUNC`
            will be used. Defaults to :py:data:`None`
        :type representation_selection_func:
            :py:class:`~collections.abc.Callable`, optional
        :param representation_selection_func_params: Parameters to obtain the
            representatives with the selection policy function. If set to
            :py:data:`None`,
            :py:attr:`~wrapper.multi_pop.DEFAULT_REPRESENTATION_SELECTION_FUNC_PARAMS`
            will be used. Defaults to :py:data:`None`
        :type representation_selection_func_params: :py:class:`dict`,
            optional
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
        :param subpop_wrapper_params: Custom parameters for the subpopulations
            (islands) wrapper
        :type subpop_wrapper_params: keyworded variable-length argument list
        :raises TypeError: If any argument is not of the appropriate type
        :raises ValueError: If any argument has an incorrect value
        """
        # Init the superclass
        super().__init__(
            fitness_function=fitness_function,
            subpop_wrapper_cls=subpop_wrapper_cls,
            num_gens=num_gens,
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
            **subpop_wrapper_params
        )

        # Get the parameters
        self.individual_cls = individual_cls
        self.species = species

    # Copy the :py:class:`culebra.wrapper.single_pop.SinglePop` properties
    individual_cls = SinglePop.individual_cls
    species = SinglePop.species

    @MultiPop.representation_topology_func.getter
    def representation_topology_func(
        self
    ) -> Callable[[int, int, Any], List[int]]:
        """Get and set the representation topology function.

        :getter: Return the representation topology function
        :setter: Set new representation topology function. If set to
            :py:data:`None`,
            :py:attr:`~wrapper.multi_pop.DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC`
            is chosen
        :type: :py:class:`~collections.abc.Callable`
        :raises TypeError: If set to a value which is not callable
        """
        return (
            DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC
            if self._representation_topology_func is None
            else self._representation_topology_func
        )

    @MultiPop.representation_topology_func_params.getter
    def representation_topology_func_params(self) -> Dict[str, Any]:
        """Get and set the parameters of the representation topology function.

        :getter: Return the current parameters for the representation topology
            function
        :setter: Set new parameters. If set to :py:data:`None`,
            :py:attr:`~wrapper.multi_pop.DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC_PARAMS`
            is chosen
        :type: :py:class:`dict`
        :raises TypeError: If set to a value which is not a :py:class:`dict`
        """
        return (
            DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC_PARAMS
            if self._representation_topology_func_params is None
            else self._representation_topology_func_params
        )

    def best_solutions(self) -> Sequence[HallOfFame]:
        """Get the best individuals found for each species.

        :return: A sequence containing :py:class:`~deap.tools.HallOfFame` of
            individuals. One hof for each species
        :rtype: :py:class:`~collections.abc.Sequence`
        :raises NotImplementedError: If has not been overriden
        """
        hof = None
        # If the search hasn't been initialized an empty HoF is returned
        if self.subpop_wrappers is None:
            hof = HallOfFame(maxsize=None)
        else:
            for subpop_wrapper in self.subpop_wrappers:
                if hof is None:
                    hof = subpop_wrapper.best_solutions()[0]
                else:
                    if subpop_wrapper.pop is not None:
                        hof.update(subpop_wrapper.pop)

        return [hof]

    @staticmethod
    def receive_representatives(subpop_wrapper) -> None:
        """Receive representative individuals.

        :param subpop_wrapper: The subpopulation wrapper receiving
            representatives
        :type subpop_wrapper: :py:class:`~wrapper.single_pop.SinglePop`
        """
        container = subpop_wrapper.container

        # Receive all the individuals in the queue
        queue = container._communication_queues[subpop_wrapper.index]
        while not queue.empty():
            subpop_wrapper._pop.extend(queue.get())

    @staticmethod
    def send_representatives(subpop_wrapper) -> None:
        """Send representatives.

        :param subpop_wrapper: The sender subpopulation wrapper
        :type subpop_wrapper: :py:class:`~wrapper.single_pop.SinglePop`
        """
        container = subpop_wrapper.container

        # Check if sending should be performed
        if subpop_wrapper._current_gen % container.representation_freq == 0:
            # Get the destinations according to the representation topology
            destinations = container.representation_topology_func(
                subpop_wrapper.index,
                container.num_subpops,
                **container.representation_topology_func_params
            )

            # For each destination
            for dest in destinations:
                # Get the representatives
                inds = container.representation_selection_func(
                    subpop_wrapper.pop,
                    container.representation_size,
                    **container.representation_selection_func_params
                )
                container._communication_queues[dest].put(inds)

    def __copy__(self) -> Islands:
        """Shallow copy the wrapper."""
        cls = self.__class__
        result = cls(
            self.individual_cls,
            self.species,
            self.fitness_function,
            self.subpop_wrapper_cls
        )
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo: dict) -> Islands:
        """Deepcopy the wrapper.

        :param memo: Wrapper attributes
        :type memo: :py:class:`dict`
        :return: A deep copy of the wrapper
        :rtype: :py:class:`~base.Wrapper`
        """
        cls = self.__class__
        result = cls(
            self.individual_cls,
            self.species,
            self.fitness_function,
            self.subpop_wrapper_cls
        )
        result.__dict__.update(deepcopy(self.__dict__, memo))
        return result

    def __reduce__(self) -> tuple:
        """Reduce the wrapper.

        :return: The reduction
        :rtype: :py:class:`tuple`
        """
        return (self.__class__,
                (
                    self.individual_cls,
                    self.species,
                    self.fitness_function,
                    self.subpop_wrapper_cls
                ),
                self.__dict__)


class HomogeneousIslands(Islands):
    """Abstract island-based model with homogeneous islands."""

    def __init__(
        self,
        individual_cls: Type[Individual],
        species: Species,
        fitness_function: FitnessFunction,
        subpop_wrapper_cls: Type[SinglePop],
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
        **subpop_wrapper_params: Any
    ) -> None:
        """Create a new wrapper.

        :param individual_cls: The individual class
        :type individual_cls: An :py:class:`~base.Individual` subclass
        :param species: The species for all the individuals
        :type species: :py:class:`~base.Species`
        :param fitness_function: The training fitness function
        :type fitness_function: :py:class:`~base.FitnessFunction`
        :param subpop_wrapper_cls: Single-population wrapper class to handle
            the subpopulations (islands).
        :type subpop_wrapper_cls: Any subclass of
            :py:class:`~wrapper.single_pop.SinglePop`
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
        :param num_subpops: The number of subpopulations (islands). If set to
            :py:data:`None`, :py:attr:`~wrapper.multi_pop.DEFAULT_NUM_SUBPOPS`
            will be used. Defaults to :py:data:`None`
        :type num_subpops: :py:class:`int`, optional
        :param representation_size: Number of representative individuals that
            will be sent to the other subpopulations. If set to
            :py:data:`None`,
            :py:attr:`~wrapper.multi_pop.DEFAULT_REPRESENTATION_SIZE` will be
            used. Defaults to :py:data:`None`
        :type representation_size: :py:class:`int`, optional
        :param representation_freq: Number of generations between
            representatives sendings. If set to :py:data:`None`,
            :py:attr:`~wrapper.multi_pop.DEFAULT_REPRESENTATION_FREQ` will be
            used. Defaults to :py:data:`None`
        :type representation_freq: :py:class:`int`, optional
        :param representation_freq: Number of generations between
            representatives sendings. If set to :py:data:`None`,
            :py:attr:`~wrapper.multi_pop.DEFAULT_REPRESENTATION_FREQ` will be
            used. Defaults to :py:data:`None`
        :type representation_freq: :py:class:`int`, optional
        :param representation_topology_func: Topology function for
            representatives sending. If set to :py:data:`None`,
            :py:attr:`~wrapper.multi_pop.DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC`
            will be used. Defaults to :py:data:`None`
        :type representation_topology_func:
            :py:class:`~collections.abc.Callable`, optional
        :param representation_topology_func_params: Parameters to obtain the
            destinations with the topology function. If set to :py:data:`None`,
            :py:attr:`~wrapper.multi_pop.DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC_PARAMS`
            will be used. Defaults to :py:data:`None`
        :type representation_topology_func_params: :py:class:`dict`, optional
        :param representation_selection_func: Policy function to choose the
            representatives from each subpopulation (island). If set to
            :py:data:`None`,
            :py:attr:`~wrapper.multi_pop.DEFAULT_REPRESENTATION_SELECTION_FUNC`
            will be used. Defaults to :py:data:`None`
        :type representation_selection_func:
            :py:class:`~collections.abc.Callable`, optional
        :param representation_selection_func_params: Parameters to obtain the
            representatives with the selection policy function. If set to
            :py:data:`None`,
            :py:attr:`~wrapper.multi_pop.DEFAULT_REPRESENTATION_SELECTION_FUNC_PARAMS`
            will be used. Defaults to :py:data:`None`
        :type representation_selection_func_params: :py:class:`dict`,
            optional
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
        :param subpop_wrapper_params: Custom parameters for the subpopulations
            (islands) wrapper
        :type subpop_wrapper_params: keyworded variable-length argument list
        :raises TypeError: If any argument is not of the appropriate type
        :raises ValueError: If any argument has an incorrect value
        """
        super().__init__(
            individual_cls=individual_cls,
            species=species,
            fitness_function=fitness_function,
            subpop_wrapper_cls=subpop_wrapper_cls,
            num_gens=num_gens,
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
            **subpop_wrapper_params
        )

        # Get the parameters
        self.pop_size = pop_size
        self.crossover_func = crossover_func
        self.mutation_func = mutation_func
        self.selection_func = selection_func
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.gene_ind_mutation_prob = gene_ind_mutation_prob
        self.selection_func_params = selection_func_params

    # Copy the :py:class:`culebra.wrapper.single_pop.SinglePop` properties
    pop_size = SinglePop.pop_size
    crossover_func = SinglePop.crossover_func
    mutation_func = SinglePop.mutation_func
    crossover_prob = SinglePop.crossover_prob
    mutation_prob = SinglePop.mutation_prob
    gene_ind_mutation_prob = SinglePop.gene_ind_mutation_prob
    selection_func = SinglePop.selection_func
    selection_func_params = SinglePop.selection_func_params

    def _generate_subpop_wrappers(self) -> None:
        """Generate the subpopulation wrappers.

        Also assign an :py:attr:`~wrapper.single_pop.SinglePop.index` and a
        :py:attr:`~wrapper.single_pop.SinglePop.container` to each
        subpopulation :py:class:`~wrapper.single_pop.SinglePop` wrapper,
        change the subpopulation wrappers'
        :py:attr:`~wrapper.single_pop.SinglePop.checkpoint_filename` according
        to the container checkpointing file name and each subpopulation index.

        Finally, the
        :py:meth:`~wrapper.single_pop.SinglePop._preprocess_generation` and
        :py:meth:`~wrapper.single_pop.SinglePop._postprocess_generation`
        methods of the
        :py:attr:`~wrapper.multi_pop.MultiPop.subpop_wrapper_cls` class are
        dynamically overriden, in order to allow individuals exchange between
        subpopulation wrappers, if necessary
        """

        def subpop_wrappers_properties() -> Dict[str, Any]:
            """Return the subpopulation wrappers' properties."""
            # Get the attributes from the container wrapper
            cls = self.subpop_wrapper_cls
            properties = {
                key: getattr(self, key)
                for key in cls.__init__.__code__.co_varnames
                if hasattr(self, key) and getattr(self, key) is not None
            }

            # Append subpopulation wrapper custom atributes
            properties.update(self.subpop_wrapper_params)

            return properties

        # Get the subpopulations properties
        properties = subpop_wrappers_properties()

        # Generate the subpopulations
        self._subpop_wrappers = []

        for (
            index,
            checkpoint_filename
        ) in enumerate(self.subpop_wrapper_checkpoint_filenames):
            subpop_wrapper = self.subpop_wrapper_cls(**properties)
            subpop_wrapper.checkpoint_filename = checkpoint_filename
            subpop_wrapper.index = index
            subpop_wrapper.container = self
            subpop_wrapper.__class__._preprocess_generation = (
                self.receive_representatives
            )
            subpop_wrapper.__class__._postprocess_generation = (
                self.send_representatives
            )
            self._subpop_wrappers.append(subpop_wrapper)


class HeterogeneousIslands(Islands):
    """Abstract island-based model with heterogeneous islands."""

    _subpop_properties_mapping = {
        "pop_sizes": "pop_size",
        "crossover_funcs": "crossover_func",
        "mutation_funcs": "mutation_func",
        "crossover_probs": "crossover_prob",
        "mutation_probs": "mutation_prob",
        "gene_ind_mutation_probs": "gene_ind_mutation_prob",
        "selection_funcs": "selection_func",
        "selection_funcs_params": "selection_func_params"
    }
    """Map the container wrapper names of properties sequences to the different
    subpopulation wrapper property names."""

    def __init__(
        self,
        individual_cls: Type[Individual],
        species: Species,
        fitness_function: FitnessFunction,
        subpop_wrapper_cls: Type[SinglePop],
        num_gens: Optional[int] = None,
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
                [Individual],
                Tuple[Individual]
            ] |
            Sequence[
                Callable[
                    [Individual],
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
        **subpop_wrapper_params: Any
    ) -> None:
        """Create a new wrapper.

        :param individual_cls: The individual class
        :type individual_cls: An :py:class:`~base.Individual` subclass
        :param species: The species for all the individuals
        :type species: :py:class:`~base.Species`
        :param fitness_function: The training fitness function
        :type fitness_function: :py:class:`~base.FitnessFunction`
        :param subpop_wrapper_cls: Single-population wrapper class to handle
            the subpopulations (islands).
        :type subpop_wrapper_cls: Any subclass of
            :py:class:`~wrapper.single_pop.SinglePop`
        :param num_gens: The number of generations. If set to
            :py:data:`None`, :py:attr:`~wrapper.DEFAULT_NUM_GENS` will
            be used. Defaults to :py:data:`None`
        :type num_gens: :py:class:`int`, optional
        :param pop_sizes: The population size for each subpopulation (island).
            If only a single value is provided, the same size will be used for
            all the subpopulations. Different sizes can be provided in a
            :py:class:`~collections.abc.Sequence`. All the sizes must be
            greater then zero. If set to :py:data:`None`,
            :py:attr:`~wrapper.single_pop.DEFAULT_POP_SIZE` will be used.
            Defaults to :py:data:`None`
        :type pop_sizes: :py:class:`int` or
            :py:class:`~collections.abc.Sequence` of :py:class:`int`, optional
        :param crossover_funcs: The crossover function for each subpopulation
            (island). If only a single value is provided, the same function
            will be used for all the subpopulations. Different functions can
            be provided in a :py:class:`~collections.abc.Sequence`. If set to
            :py:data:`None`, the *individual_cls*
            :py:meth:`~base.Individual.crossover` method will be used.
            Defaults to :py:data:`None`
        :type crossover_funcs: :py:class:`~collections.abc.Callable` or
            :py:class:`~collections.abc.Sequence` of
            :py:class:`~collections.abc.Callable`, optional
        :param mutation_funcs: The mutation function for each subpopulation
            (island). If only a single value is provided, the same function
            will be used for all the subpopulations. Different functions can
            be provided in a :py:class:`~collections.abc.Sequence`. If set to
            :py:data:`None`, the *individual_cls*
            :py:meth:`~base.Individual.mutate` method will be used.
            Defaults to :py:data:`None`
        :type mutation_funcs: :py:class:`~collections.abc.Callable` or
            :py:class:`~collections.abc.Sequence` of
            :py:class:`~collections.abc.Callable`, optional
        :param selection_funcs: The selection function for each subpopulation
            (island). If only a single value is provided, the same function
            will be used for all the subpopulations. Different functions can
            be provided in a :py:class:`~collections.abc.Sequence`. If set to
            :py:data:`None`,
            :py:attr:`~wrapper.single_pop.DEFAULT_SELECTION_FUNC` will be
            used. Defaults to :py:data:`None`
        :type selection_funcs: :py:class:`~collections.abc.Callable` or
            :py:class:`~collections.abc.Sequence` of
            :py:class:`~collections.abc.Callable`, optional
        :param crossover_probs: The crossover probability for each
            subpopulation (island). If only a single value is provided, the
            same probability will be used for all the subpopulations.
            Different probabilities can be provided in a
            :py:class:`~collections.abc.Sequence`. All the probabilities must
            be in (0, 1). If set to :py:data:`None`,
            :py:attr:`~wrapper.single_pop.DEFAULT_CROSSOVER_PROB` will be
            used. Defaults to :py:data:`None`
        :type crossover_probs: :py:class:`float` or
            :py:class:`~collections.abc.Sequence` of :py:class:`float`,
            optional
        :param mutation_probs: The mutation probability for each subpopulation
            (island). If only a single value is provided, the same probability
            will be used for all the subpopulations. Different probabilities
            can be provided in a :py:class:`~collections.abc.Sequence`. All
            the probabilities must be in (0, 1). If set to :py:data:`None`,
            :py:attr:`~wrapper.single_pop.DEFAULT_MUTATION_PROB` will be used.
            Defaults to :py:data:`None`
        :type mutation_probs: :py:class:`float` or
            :py:class:`~collections.abc.Sequence` of :py:class:`float`,
            optional
        :param gene_ind_mutation_probs: The gene independent mutation
            probability for each subpopulation (island). If only a single
            value is provided, the same probability will be used for all the
            subpopulations. Different probabilities can be provided in a
            :py:class:`~collections.abc.Sequence`. All the probabilities must
            be in (0, 1). If set to :py:data:`None`,
            :py:attr:`~wrapper.single_pop.DEFAULT_GENE_IND_MUTATION_PROB` will
            be used. Defaults to :py:data:`None`
        :type gene_ind_mutation_probs: :py:class:`float` or
            :py:class:`~collections.abc.Sequence` of :py:class:`float`,
            optional
        :param selection_funcs_params: The parameters for the selection
            function of each subpopulation (island). If only a single value is
            provided, the same probability will be used for all the
            subpopulations. Different probabilities can be provided in a
            :py:class:`~collections.abc.Sequence`. If set to :py:data:`None`,
            :py:attr:`~wrapper.single_pop.DEFAULT_SELECTION_FUNC_PARAMS` will
            be used. Defaults to :py:data:`None`
        :type selection_funcs_params: :py:class:`dict` or
            :py:class:`~collections.abc.Sequence` of :py:class:`dict`, optional
        :param num_subpops: The number of subpopulations (islands). If set to
            :py:data:`None`, :py:attr:`~wrapper.multi_pop.DEFAULT_NUM_SUBPOPS`
            will be used. Defaults to :py:data:`None`
        :type num_subpops: :py:class:`int`, optional
        :param representation_size: Number of representative individuals that
            will be sent to the other subpopulations. If set to
            :py:data:`None`,
            :py:attr:`~wrapper.multi_pop.DEFAULT_REPRESENTATION_SIZE` will be
            used. Defaults to :py:data:`None`
        :type representation_size: :py:class:`int`, optional
        :param representation_freq: Number of generations between
            representatives sendings. If set to :py:data:`None`,
            :py:attr:`~wrapper.multi_pop.DEFAULT_REPRESENTATION_FREQ` will be
            used. Defaults to :py:data:`None`
        :type representation_freq: :py:class:`int`, optional
        :param representation_freq: Number of generations between
            representatives sendings. If set to :py:data:`None`,
            :py:attr:`~wrapper.multi_pop.DEFAULT_REPRESENTATION_FREQ` will be
            used. Defaults to :py:data:`None`
        :type representation_freq: :py:class:`int`, optional
        :param representation_topology_func: Topology function for
            representatives sending. If set to :py:data:`None`,
            :py:attr:`~wrapper.multi_pop.DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC`
            will be used. Defaults to :py:data:`None`
        :type representation_topology_func:
            :py:class:`~collections.abc.Callable`, optional
        :param representation_topology_func_params: Parameters to obtain the
            destinations with the topology function. If set to :py:data:`None`,
            :py:attr:`~wrapper.multi_pop.DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC_PARAMS`
            will be used. Defaults to :py:data:`None`
        :type representation_topology_func_params: :py:class:`dict`, optional
        :param representation_selection_func: Policy function to choose the
            representatives from each subpopulation (island). If set to
            :py:data:`None`,
            :py:attr:`~wrapper.multi_pop.DEFAULT_REPRESENTATION_SELECTION_FUNC`
            will be used. Defaults to :py:data:`None`
        :type representation_selection_func:
            :py:class:`~collections.abc.Callable`, optional
        :param representation_selection_func_params: Parameters to obtain the
            representatives with the selection policy function. If set to
            :py:data:`None`,
            :py:attr:`~wrapper.multi_pop.DEFAULT_REPRESENTATION_SELECTION_FUNC_PARAMS`
            will be used. Defaults to :py:data:`None`
        :type representation_selection_func_params: :py:class:`dict`,
            optional
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
        :param subpop_wrapper_params: Custom parameters for the subpopulations
            (islands) wrapper
        :type subpop_wrapper_params: keyworded variable-length argument list
        :raises TypeError: If any argument is not of the appropriate type
        :raises ValueError: If any argument has an incorrect value
        """
        super().__init__(
            individual_cls=individual_cls,
            species=species,
            fitness_function=fitness_function,
            subpop_wrapper_cls=subpop_wrapper_cls,
            num_gens=num_gens,
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
            **subpop_wrapper_params
        )

        # Get the parameters
        self.pop_sizes = pop_sizes
        self.crossover_funcs = crossover_funcs
        self.mutation_funcs = mutation_funcs
        self.selection_funcs = selection_funcs
        self.crossover_probs = crossover_probs
        self.mutation_probs = mutation_probs
        self.gene_ind_mutation_probs = gene_ind_mutation_probs
        self.selection_funcs_params = selection_funcs_params

    @property
    def pop_sizes(self) -> Sequence[int | None]:
        """Get and set the population size for each subpopulation.

        :getter: Return the current size of each subpopulation
        :setter: Set a new size for each subpopulation. If only a single value
            is provided, the same size will be used for all the subpopulations.
            Different sizes can be provided in a
            :py:class:`~collections.abc.Sequence`. All the sizes must be
            greater then zero. If set to :py:data:`None`,
            :py:attr:`~wrapper.single_pop.DEFAULT_POP_SIZE` is chosen
        :type: :py:class:`int` or :py:class:`~collections.abc.Sequence`
            of :py:class:`int`
        :raises TypeError: If set to a value which is not an :py:class:`int`
            or a :py:class:`~collections.abc.Sequence` of :py:class:`int`
        :raises ValueError: If any population size is not greater than zero
        """
        if self.subpop_wrappers is not None:
            the_pop_sizes = [
                subpop_wrapper.pop_size
                for subpop_wrapper in self.subpop_wrappers
            ]
        elif isinstance(self._pop_sizes, Sequence):
            the_pop_sizes = self._pop_sizes
        else:
            the_pop_sizes = list(repeat(self._pop_sizes, self.num_subpops))

        return the_pop_sizes

    @pop_sizes.setter
    def pop_sizes(self, sizes: int | Sequence[int] | None) -> None:
        """Set the population size for each subpopulation.

        :param sizes: The new population sizes. If only a single value
            is provided, the same size will be used for all the subpopulations.
            Different sizes can be provided in a
            :py:class:`~collections.abc.Sequence`. All the sizes must be
            greater then zero. If set to :py:data:`None`,
        :py:attr:`~wrapper.single_pop.DEFAULT_POP_SIZE` is chosen
        :type: :py:class:`int` or :py:class:`~collections.abc.Sequence`
            of :py:class:`int`
        :raises TypeError: If *sizes* is not an :py:class:`int`
            or a :py:class:`~collections.abc.Sequence` of :py:class:`int`
        :raises ValueError: If any value in *size* is not greater than zero
        """
        # If None is provided ...
        if sizes is None:
            self._pop_sizes = None
        # If a sequence is provided ...
        elif isinstance(sizes, Sequence):
            self._pop_sizes = check_sequence(
                sizes,
                "population sizes",
                item_checker=partial(check_int, gt=0)
            )
        # If a scalar value is provided ...
        else:
            self._pop_sizes = check_int(sizes, "population size", gt=0)

        # Reset the algorithm
        self.reset()

    @property
    def crossover_funcs(self) -> Sequence[
        Callable[[Individual, Individual], Tuple[Individual, Individual]] |
        None
    ]:
        """Get and set the crossover function for each subpopulation.

        :getter: Return the current crossover functions
        :setter: Set the new crossover functions. If only a single value is
            provided, the same function will be used for all the
            subpopulations. Different functions can be provided in a
            :py:class:`~collections.abc.Sequence`. If set to :py:data:`None`,
            the :py:meth:`~base.Individual.crossover` method of the individual
            class evolved by the wrapper is chosen
        :type: :py:class:`~collections.abc.Callable` or a
            :py:class:`~collections.abc.Sequence` of
            :py:class:`~collections.abc.Callable`
        :raises TypeError: If set to a value which is not
            :py:class:`~collections.abc.Callable` or a
            :py:class:`~collections.abc.Sequence` of
            :py:class:`~collections.abc.Callable`
        """
        if self.subpop_wrappers is not None:
            the_funcs = [
                subpop_wrapper.crossover_func
                for subpop_wrapper in self.subpop_wrappers
            ]
        elif isinstance(self._crossover_funcs, Sequence):
            the_funcs = self._crossover_funcs
        else:
            the_funcs = list(repeat(self._crossover_funcs, self.num_subpops))

        return the_funcs

    @crossover_funcs.setter
    def crossover_funcs(
        self,
        funcs: Callable[
            [Individual, Individual],
            Tuple[Individual, Individual]
        ] | Sequence[
            Callable[
                [Individual, Individual],
                Tuple[Individual, Individual]
            ]
        ] | None
    ) -> None:
        """Set the crossover function for each subpopulation.

        :param funcs: The new crossover functions. If only a single value is
            provided, the same function will be used for all the
            subpopulations. Different functions can be provided in a
            :py:class:`~collections.abc.Sequence`. If set to :py:data:`None`,
            the :py:meth:`~base.Individual.crossover` method of the individual
            class evolved by the wrapper is chosen
        :type funcs: :py:class:`~collections.abc.Callable` or a
            :py:class:`~collections.abc.Sequence` of
            :py:class:`~collections.abc.Callable`
        :raises TypeError: If *funcs* is not
            :py:class:`~collections.abc.Callable` or a
            :py:class:`~collections.abc.Sequence` of
            :py:class:`~collections.abc.Callable`
        """
        if funcs is None:
            self._crossover_funcs = None
        elif isinstance(funcs, Sequence):
            self._crossover_funcs = check_sequence(
                funcs,
                "crossover functions",
                item_checker=check_func
            )
        else:
            self._crossover_funcs = check_func(funcs, "crossover function")

        # Reset the algorithm
        self.reset()

    @property
    def mutation_funcs(self) -> Sequence[
        Callable[[Individual], Tuple[Individual]] | None
    ]:
        """Get and set the mutation function for each subpopulation.

        :getter: Return the current mutation functions
        :setter: Set the new mutation functions. If only a single value is
            provided, the same function will be used for all the
            subpopulations. Different functions can be provided in a
            :py:class:`~collections.abc.Sequence`. If set to :py:data:`None`,
            the :py:meth:`~base.Individual.mutate` method of the individual
            class evolved by the wrapper is chosen
        :type: :py:class:`~collections.abc.Callable` or a
            :py:class:`~collections.abc.Sequence` of
            :py:class:`~collections.abc.Callable`
        :raises TypeError: If set to a value which is not
            :py:class:`~collections.abc.Callable` or a
            :py:class:`~collections.abc.Sequence` of
            :py:class:`~collections.abc.Callable`
        """
        if self.subpop_wrappers is not None:
            the_funcs = [
                subpop_wrapper.mutation_func
                for subpop_wrapper in self.subpop_wrappers
            ]
        elif isinstance(self._mutation_funcs, Sequence):
            the_funcs = self._mutation_funcs
        else:
            the_funcs = list(repeat(self._mutation_funcs, self.num_subpops))

        return the_funcs

    @mutation_funcs.setter
    def mutation_funcs(
        self,
        funcs: Callable[
            [Individual],
            Tuple[Individual]
        ] | Sequence[
            Callable[
                [Individual],
                Tuple[Individual]
            ]
        ] | None
    ) -> None:
        """Set the mutation function for each subpopulation.

        :param funcs: The new mutation functions. If only a single value is
            provided, the same function will be used for all the
            subpopulations. Different functions can be provided in a
            :py:class:`~collections.abc.Sequence`. If set to :py:data:`None`,
            the :py:meth:`~base.Individual.mutate` method of the individual
            class evolved by the wrapper is chosen
        :type funcs: :py:class:`~collections.abc.Callable` or a
            :py:class:`~collections.abc.Sequence` of
            :py:class:`~collections.abc.Callable`
        :raises TypeError: If *funcs* is not
            :py:class:`~collections.abc.Callable` or a
            :py:class:`~collections.abc.Sequence` of
            :py:class:`~collections.abc.Callable`
        """
        if funcs is None:
            self._mutation_funcs = None
        elif isinstance(funcs, Sequence):
            self._mutation_funcs = check_sequence(
                funcs,
                "mutation functions",
                item_checker=check_func
            )
        else:
            self._mutation_funcs = check_func(funcs, "mutation function")

        # Reset the algorithm
        self.reset()

    @property
    def crossover_probs(self) -> Sequence[float | None]:
        """Get and set the crossover probability for each subpopulation.

        :getter: Return the current crossover probabilities
        :setter: Set the new crossover probabilities. If only a single value
            is provided, the same probability will be used for all the
            subpopulations. Different probabilities can be provided in a
            :py:class:`~collections.abc.Sequence`. All the probabilities must
            be in (0, 1). If set to :py:data:`None`,
            :py:attr:`~wrapper.single_pop.DEFAULT_CROSSOVER_PROB` is chosen
        :type: :py:class:`float` or :py:class:`~collections.abc.Sequence` of
            :py:class:`float`
        :raises TypeError: If set to a value which is not a real number or a
            :py:class:`~collections.abc.Sequence` of real numbers
        :raises ValueError: If any probability is not in (0, 1)
        """
        if self.subpop_wrappers is not None:
            the_probs = [
                subpop_wrapper.crossover_prob
                for subpop_wrapper in self.subpop_wrappers
            ]
        elif isinstance(self._crossover_probs, Sequence):
            the_probs = self._crossover_probs
        else:
            the_probs = list(repeat(self._crossover_probs, self.num_subpops))

        return the_probs

    @crossover_probs.setter
    def crossover_probs(self, probs: float | Sequence[float] | None) -> None:
        """Set the crossover probability for each subpopulation.

        :param probs: The new probabilities. If only a single value
            is provided, the same probability will be used for all the
            subpopulations. Different probabilities can be provided in a
            :py:class:`~collections.abc.Sequence`. All the probabilities must
            be in (0, 1). If set to :py:data:`None`,
            :py:attr:`~wrapper.single_pop.DEFAULT_CROSSOVER_PROB` is chosen
        :type: :py:class:`float` or :py:class:`~collections.abc.Sequence` of
            :py:class:`float`
        :raises TypeError: If *probs* is not a real number or a
            :py:class:`~collections.abc.Sequence` of real numbers
        :raises ValueError: If any probability is not in (0, 1)
        """
        if probs is None:
            self._crossover_probs = None
        # If a sequence is provided ...
        elif isinstance(probs, Sequence):
            self._crossover_probs = check_sequence(
                probs,
                "crossover probabilities",
                item_checker=partial(check_float, gt=0, lt=1)
            )
        else:
            self._crossover_probs = check_float(
                probs, "crossover probability", gt=0, lt=1
            )

        # Reset the algorithm
        self.reset()

    @property
    def mutation_probs(self) -> Sequence[float | None]:
        """Get and set the mutation probability for each subpopulation.

        :getter: Return the current mutation probabilities
        :setter: Set the new mutation probabilities. If only a single value
            is provided, the same probability will be used for all the
            subpopulations. Different probabilities can be provided in a
            :py:class:`~collections.abc.Sequence`. All the probabilities must
            be in (0, 1). If set to :py:data:`None`,
            :py:attr:`~wrapper.single_pop.DEFAULT_MUTATION_PROB` is chosen
        :type: :py:class:`float` or :py:class:`~collections.abc.Sequence` of
            :py:class:`float`
        :raises TypeError: If set to a value which is not a real number or a
            :py:class:`~collections.abc.Sequence` of real numbers
        :raises ValueError: If probabilities are not in (0, 1)
        """
        if self.subpop_wrappers is not None:
            the_probs = [
                subpop_wrapper.mutation_prob
                for subpop_wrapper in self.subpop_wrappers
            ]
        elif isinstance(self._mutation_probs, Sequence):
            the_probs = self._mutation_probs
        else:
            the_probs = list(repeat(self._mutation_probs, self.num_subpops))

        return the_probs

    @mutation_probs.setter
    def mutation_probs(self, probs: float | Sequence[float] | None):
        """Set the mutation probability for each subpopulation.

        :param probs: The new probabilities. If only a single value
            is provided, the same probability will be used for all the
            subpopulations. Different probabilities can be provided in a
            :py:class:`~collections.abc.Sequence`. All the probabilities must
            be in (0, 1). If set to :py:data:`None`,
            :py:attr:`~wrapper.single_pop.DEFAULT_MUTATION_PROB` is chosen
        :type: :py:class:`float` or :py:class:`~collections.abc.Sequence` of
            :py:class:`float`
        :raises TypeError: If *probs* is not a real number or a
            :py:class:`~collections.abc.Sequence` of real numbers
        :raises ValueError: If probabilities are not in (0, 1)
        """
        if probs is None:
            self._mutation_probs = None
        # If a sequence is provided ...
        elif isinstance(probs, Sequence):
            self._mutation_probs = check_sequence(
                probs,
                "mutation probabilities",
                item_checker=partial(check_float, gt=0, lt=1)
            )
        else:
            self._mutation_probs = check_float(
                probs, "mutation probability", gt=0, lt=1
            )

        # Reset the algorithm
        self.reset()

    @property
    def gene_ind_mutation_probs(self) -> Sequence[float | None]:
        """Get and set the gene independent mutation probabilities.

        :getter: Return the current gene independent mutation probability for
            each subpopulation
        :setter: Set new values for the gene independent mutation
            probabilities. If only a single value is provided, the same
            probability will be used for all the subpopulations. Different
            probabilities can be provided in a
            :py:class:`~collections.abc.Sequence`. All the probabilities must
            be in (0, 1). If set to :py:data:`None`,
            :py:attr:`~wrapper.single_pop.DEFAULT_GENE_IND_MUTATION_PROB` is
            chosen
        :type: :py:class:`float` or :py:class:`~collections.abc.Sequence` of
            :py:class:`float`
        :raises TypeError: If set to a value which is not a real number or a
            :py:class:`~collections.abc.Sequence` of real numbers
        :raises ValueError: If probabilities are not in (0, 1)
        """
        if self.subpop_wrappers is not None:
            the_probs = [
                subpop_wrapper.gene_ind_mutation_prob
                for subpop_wrapper in self.subpop_wrappers
            ]
        elif isinstance(self._gene_ind_mutation_probs, Sequence):
            the_probs = self._gene_ind_mutation_probs
        else:
            the_probs = list(
                repeat(self._gene_ind_mutation_probs, self.num_subpops)
            )

        return the_probs

    @gene_ind_mutation_probs.setter
    def gene_ind_mutation_probs(self, probs: float | Sequence[float] | None):
        """Set the subpopulations gene independent mutation probability.

        :param probs: The new probabilities. If only a single value is
            provided, the same probability will be used for all the
            subpopulations. Different probabilities can be provided in a
            :py:class:`~collections.abc.Sequence`. All the probabilities must
            be in (0, 1). If set to :py:data:`None`,
            :py:attr:`~wrapper.single_pop.DEFAULT_GENE_IND_MUTATION_PROB` is
            chosen
        :type: :py:class:`float` or :py:class:`~collections.abc.Sequence` of
            :py:class:`float`
        :raises TypeError: If *probs* is not a real number or a
            :py:class:`~collections.abc.Sequence` of real numbers
        :raises ValueError: If probabilities are not in (0, 1)
        """
        if probs is None:
            self._gene_ind_mutation_probs = None
        # If a sequence is provided ...
        elif isinstance(probs, Sequence):
            self._gene_ind_mutation_probs = check_sequence(
                probs,
                "gene independent mutation probabilities",
                item_checker=partial(check_float, gt=0, lt=1)
            )
        else:
            self._gene_ind_mutation_probs = check_float(
                probs,
                "gene independent mutation probability",
                gt=0,
                lt=1
            )

        # Reset the algorithm
        self.reset()

    @property
    def selection_funcs(self) -> Sequence[
        Callable[[List[Individual], int, Any], List[Individual]] | None
    ]:
        """Get and set the selection function for each subpopulation.

        :getter: Return the current selection functions
        :setter: Set the new selection functions. If only a single value is
            provided, the same function will be used for all the
            subpopulations. Different functions can be provided in a
            :py:class:`~collections.abc.Sequence`. If set to :py:data:`None`,
            :py:attr:`~wrapper.single_pop.DEFAULT_SELECTION_FUNC` is chosen
        :type: :py:class:`~collections.abc.Callable` or a
            :py:class:`~collections.abc.Sequence` of
            :py:class:`~collections.abc.Callable`
        :raises TypeError: If set to a value which is not
            :py:class:`~collections.abc.Callable` or a
            :py:class:`~collections.abc.Sequence` of
            :py:class:`~collections.abc.Callable`
        """
        if self.subpop_wrappers is not None:
            the_funcs = [
                subpop_wrapper.selection_func
                for subpop_wrapper in self.subpop_wrappers
            ]
        elif isinstance(self._selection_funcs, Sequence):
            the_funcs = self._selection_funcs
        else:
            the_funcs = list(repeat(self._selection_funcs, self.num_subpops))

        return the_funcs

    @selection_funcs.setter
    def selection_funcs(self, funcs: Callable[
            [List[Individual], int, Any],
            List[Individual]
        ] | Sequence[
            Callable[
                [List[Individual], int, Any],
                List[Individual]
            ]
    ] | None
    ) -> None:
        """Set the selection function for each subpopulation.

        :param funcs: The new selection functions. If only a single value is
            provided, the same function will be used for all the
            subpopulations. Different functions can be provided in a
            :py:class:`~collections.abc.Sequence`. If set to :py:data:`None`,
            :py:attr:`~wrapper.single_pop.DEFAULT_SELECTION_FUNC` is chosen
        :type funcs: :py:class:`~collections.abc.Callable` or a
            :py:class:`~collections.abc.Sequence` of
            :py:class:`~collections.abc.Callable`
        :raises TypeError: If *funcs* is not
            :py:class:`~collections.abc.Callable` or a
            :py:class:`~collections.abc.Sequence` of
            :py:class:`~collections.abc.Callable`
        """
        if funcs is None:
            self._selection_funcs = None
        # If a sequence is provided ...
        elif isinstance(funcs, Sequence):
            self._selection_funcs = check_sequence(
                funcs,
                "selection functions",
                item_checker=check_func
            )
        else:
            self._selection_funcs = check_func(
                funcs,
                "selection function"
            )

        # Reset the algorithm
        self.reset()

    @property
    def selection_funcs_params(self) -> Sequence[Dict[str, Any] | None]:
        """Get and set the parameters of the selection functions.

        :getter: Return the current parameters for the selection function of
            each subpopulation
        :setter: Set new parameters. If only a single value is provided, the
            same probability will be used for all the subpopulations.
            Different probabilities can be provided in a
            :py:class:`~collections.abc.Sequence`. If set to :py:data:`None`,
            :py:attr:`~wrapper.single_pop.DEFAULT_SELECTION_FUNC_PARAMS` is
            chosen
        :type: :py:class:`dict` or :py:class:`~collections.abc.Sequence`
            of :py:class:`dict`
        :raises TypeError: If set to a value which is not a :py:class:`dict`
            or a :py:class:`~collections.abc.Sequence`
            of :py:class:`dict`
        :raises ValueError: If any element of the sequence is not a
            :py:class:`dict`
        """
        if self.subpop_wrappers is not None:
            the_params = [
                subpop_wrapper.selection_func_params
                for subpop_wrapper in self.subpop_wrappers
            ]
        elif isinstance(self._selection_funcs_params, Sequence):
            the_params = self._selection_funcs_params
        else:
            the_params = list(
                repeat(self._selection_funcs_params, self.num_subpops)
            )

        return the_params

    @selection_funcs_params.setter
    def selection_funcs_params(
        self,
        param_dicts: Dict[str, Any] | Sequence[Dict[str, Any]] | None
    ) -> None:
        """Set the parameters for the selection function of each subpopulation.

        :param param_dicts: The new parameters. If only a single value is
            provided, the same probability will be used for all the
            subpopulations. Different probabilities can be provided in a
            :py:class:`~collections.abc.Sequence`. If set to :py:data:`None`,
            :py:attr:`~wrapper.single_pop.DEFAULT_SELECTION_FUNC_PARAMS` is
            chosen
        :type param_dicts: A :py:class:`dict` or a
            :py:class:`~collections.abc.Sequence` of :py:class:`dict`
        :raises TypeError: If *param_dicts* is not a :py:class:`dict`
            or a :py:class:`~collections.abc.Sequence`
        :raises ValueError: If any element of the sequence is not a
            :py:class:`dict`
        """
        if param_dicts is None:
            self._selection_funcs_params = None
        # If a sequence is provided ...
        elif isinstance(param_dicts, Sequence):
            self._selection_funcs_params = check_sequence(
                param_dicts,
                "selection functions parameters",
                item_checker=check_func_params
            )
        else:
            self._selection_funcs_params = check_func_params(
                param_dicts,
                "selection function parameters"
            )

        # Reset the algorithm
        self.reset()

    def _generate_subpop_wrappers(self) -> None:
        """Generate the subpopulation wrappers.

        Also assign an :py:attr:`~wrapper.single_pop.SinglePop.index` and a
        :py:attr:`~wrapper.single_pop.SinglePop.container` to each
        subpopulation :py:class:`~wrapper.single_pop.SinglePop` wrapper,
        change the subpopulation wrappers'
        :py:attr:`~wrapper.single_pop.SinglePop.checkpoint_filename` according
        to the container checkpointing file name and each subpopulation index.

        Finally, the
        :py:meth:`~wrapper.single_pop.SinglePop._preprocess_generation` and
        :py:meth:`~wrapper.single_pop.SinglePop._postprocess_generation`
        methods of the
        :py:attr:`~wrapper.multi_pop.MultiPop.subpop_wrapper_cls` class are
        dynamically overriden, in order to allow individuals exchange between
        subpopulation wrappers, if necessary

        :raises RuntimeError: If the length of any properties sequence does
            not match the number of subpopulations.
        """

        def subpop_wrappers_properties() -> List[Dict[str, Any]]:
            """Obtain the properties of each subpopulation wrapper.

            :raises RuntimeError: If the length of any properties sequence
                does not match the number of subpopulations.

            :return: The properties of each subpopulation wrapper.
            :rtype: :py:class:`list`
            """
            # Get the common attributes from the container wrapper
            cls = self.subpop_wrapper_cls
            common_properties = {
                key: getattr(self, key)
                for key in cls.__init__.__code__.co_varnames
                if hasattr(self, key) and getattr(self, key) is not None
            }

            # Append subpopulation wrapper custom atributes
            common_properties.update(self.subpop_wrapper_params)

            # List with the common properties. Equal for all the subpopulations
            properties = []
            for _ in range(self.num_subpops):
                subpop_properties = {}
                for key, value in common_properties.items():
                    subpop_properties[key] = value
                properties.append(subpop_properties)

            # Particular properties for each subpop
            cls = self.__class__
            for (
                    property_sequence_name,
                    subpop_property_name
            ) in self._subpop_properties_mapping.items():

                # Values of the sequence
                property_sequence_values = getattr(
                    cls, property_sequence_name
                ).fget(self)

                # Check the properties' length
                if len(property_sequence_values) != self.num_subpops:
                    raise RuntimeError(
                        f"The length of {property_sequence_name} does not "
                        "match the number of subpopulations"
                    )
                for (
                        subpop_properties, subpop_property_value
                ) in zip(properties, property_sequence_values):
                    subpop_properties[
                        subpop_property_name] = subpop_property_value

            return properties

        # Get the subpopulations properties
        properties = subpop_wrappers_properties()

        # Generate the subpopulations
        self._subpop_wrappers = []

        for (
            index, (
                checkpoint_filename,
                subpop_properties
            )
        ) in enumerate(
            zip(self.subpop_wrapper_checkpoint_filenames, properties)
        ):
            subpop_wrapper = self.subpop_wrapper_cls(**subpop_properties)
            subpop_wrapper.checkpoint_filename = checkpoint_filename
            subpop_wrapper.index = index
            subpop_wrapper.container = self
            subpop_wrapper.__class__._preprocess_generation = (
                self.receive_representatives
            )
            subpop_wrapper.__class__._postprocess_generation = (
                self.send_representatives
            )
            self._subpop_wrappers.append(subpop_wrapper)


class HomogeneousSequentialIslands(HomogeneousIslands, SequentialMultiPop):
    """Sequential island-based model with homogeneous islands."""

    def __init__(
        self,
        individual_cls: Type[Individual],
        species: Species,
        fitness_function: FitnessFunction,
        subpop_wrapper_cls: Type[SinglePop],
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
        **subpop_wrapper_params: Any
    ) -> None:
        """."""
        HomogeneousIslands.__init__(
            self,
            individual_cls,
            species,
            fitness_function,
            subpop_wrapper_cls,
            pop_size=pop_size,
            crossover_func=crossover_func,
            mutation_func=mutation_func,
            selection_func=selection_func,
            crossover_prob=crossover_prob,
            mutation_prob=mutation_prob,
            gene_ind_mutation_prob=gene_ind_mutation_prob,
            selection_func_params=selection_func_params
        )

        SequentialMultiPop.__init__(
            self,
            fitness_function,
            subpop_wrapper_cls,
            num_gens=num_gens,
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
            **subpop_wrapper_params
        )

    # __init__ shares the documentation with HomogeneousIslands.__init__
    __init__.__doc__ = HomogeneousIslands.__init__.__doc__


class HomogeneousParallelIslands(HomogeneousIslands, ParallelMultiPop):
    """Parallel island-based model with homogeneous islands."""

    def __init__(
        self,
        individual_cls: Type[Individual],
        species: Species,
        fitness_function: FitnessFunction,
        subpop_wrapper_cls: Type[SinglePop],
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
        **subpop_wrapper_params: Any
    ) -> None:
        """."""
        # Call both constructors
        HomogeneousIslands.__init__(
            self,
            individual_cls,
            species,
            fitness_function,
            subpop_wrapper_cls,
            pop_size=pop_size,
            crossover_func=crossover_func,
            mutation_func=mutation_func,
            selection_func=selection_func,
            crossover_prob=crossover_prob,
            mutation_prob=mutation_prob,
            gene_ind_mutation_prob=gene_ind_mutation_prob,
            selection_func_params=selection_func_params
        )

        ParallelMultiPop.__init__(
            self,
            fitness_function,
            subpop_wrapper_cls,
            num_gens=num_gens,
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
            **subpop_wrapper_params
        )

    # Change the docstring of the constructor to indicate that the default
    # number of subpopulations is the number of CPU cores for parallel
    # multi-population approaches
    __init__.__doc__ = HomogeneousIslands.__init__.__doc__.replace(
        ':py:attr:`~wrapper.multi_pop.DEFAULT_NUM_SUBPOPS`',
        'the number of CPU cores'
    )


class HeterogeneousSequentialIslands(HeterogeneousIslands, SequentialMultiPop):
    """Sequential island-based model with heterogeneous islands."""

    def __init__(
        self,
        individual_cls: Type[Individual],
        species: Species,
        fitness_function: FitnessFunction,
        subpop_wrapper_cls: Type[SinglePop],
        num_gens: Optional[int] = None,
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
                [Individual],
                Tuple[Individual]
            ] |
            Sequence[
                Callable[
                    [Individual],
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
        **subpop_wrapper_params: Any
    ) -> None:
        """."""
        # Call both constructors
        HeterogeneousIslands.__init__(
            self,
            individual_cls,
            species,
            fitness_function,
            subpop_wrapper_cls,
            num_subpops=num_subpops,
            pop_sizes=pop_sizes,
            crossover_funcs=crossover_funcs,
            mutation_funcs=mutation_funcs,
            selection_funcs=selection_funcs,
            crossover_probs=crossover_probs,
            mutation_probs=mutation_probs,
            gene_ind_mutation_probs=gene_ind_mutation_probs,
            selection_funcs_params=selection_funcs_params
        )

        SequentialMultiPop.__init__(
            self,
            fitness_function,
            subpop_wrapper_cls,
            num_gens=num_gens,
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
            **subpop_wrapper_params
        )

    # __init__ shares the documentation with HeterogeneousIslands.__init__
    __init__.__doc__ = HeterogeneousIslands.__init__.__doc__


class HeterogeneousParallelIslands(HeterogeneousIslands, ParallelMultiPop):
    """Parallel island-based model with heterogeneous islands."""

    def __init__(
        self,
        individual_cls: Type[Individual],
        species: Species,
        fitness_function: FitnessFunction,
        subpop_wrapper_cls: Type[SinglePop],
        num_gens: Optional[int] = None,
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
                [Individual],
                Tuple[Individual]
            ] |
            Sequence[
                Callable[
                    [Individual],
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
        **subpop_wrapper_params: Any
    ) -> None:
        """."""
        # Call both constructors
        HeterogeneousIslands.__init__(
            self,
            individual_cls,
            species,
            fitness_function,
            subpop_wrapper_cls,
            num_subpops=num_subpops,
            pop_sizes=pop_sizes,
            crossover_funcs=crossover_funcs,
            mutation_funcs=mutation_funcs,
            selection_funcs=selection_funcs,
            crossover_probs=crossover_probs,
            mutation_probs=mutation_probs,
            gene_ind_mutation_probs=gene_ind_mutation_probs,
            selection_funcs_params=selection_funcs_params
        )

        ParallelMultiPop.__init__(
            self,
            fitness_function,
            subpop_wrapper_cls,
            num_gens=num_gens,
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
            **subpop_wrapper_params
        )

    # Change the docstring of the constructor to indicate that the default
    # number of subpopulations is the number of CPU cores for parallel
    # multi-population approaches
    __init__.__doc__ = HeterogeneousIslands.__init__.__doc__.replace(
        ':py:attr:`~wrapper.multi_pop.DEFAULT_NUM_SUBPOPS`',
        'the number of CPU cores'
    )


__all__ = [
    'Islands',
    'HomogeneousIslands',
    'HeterogeneousIslands',
    'HomogeneousSequentialIslands',
    'HomogeneousParallelIslands',
    'HeterogeneousSequentialIslands',
    'HeterogeneousParallelIslands',
    'DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC',
    'DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC_PARAMS',
]
