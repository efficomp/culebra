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

"""Cooperative co-evolutionary wrapper.

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
    Sequence)
from copy import deepcopy
from functools import partial, partialmethod
from deap.tools import HallOfFame
from culebra.base import (
    Individual,
    Species,
    FitnessFunction,
    check_instance,
    check_subclass,
    check_sequence
)
from culebra.wrapper.single_pop import SinglePop
from . import (
    MultiPop,
    SequentialMultiPop,
    ParallelMultiPop,
    HeterogeneousIslands,
    DEFAULT_REPRESENTATION_TOPOLOGY_FUNC,
    DEFAULT_REPRESENTATION_TOPOLOGY_FUNC_PARAMS
)


__author__ = "Jesús González"
__copyright__ = "Copyright 2021, EFFICOMP"
__license__ = "GNU GPL-3.0-or-later"
__version__ = "0.1.1"
__maintainer__ = "Jesús González"
__email__ = "jesusgonzalez@ugr.es"
__status__ = "Development"


class Cooperative(MultiPop):
    """Abstract cooperative co-evolutionary model for wrappers."""

    _subpop_properties_mapping = {
        "individual_classes": "individual_cls",
        "species": "species",
        "pop_sizes": "pop_size",
        "crossover_funcs": "crossover_func",
        "mutation_funcs": "mutation_func",
        "crossover_probs": "crossover_prob",
        "mutation_probs": "mutation_prob",
        "gene_ind_mutation_probs": "gene_ind_mutation_prob",
        "selection_funcs": "selection_func",
        "selection_funcs_params": "selection_func_params"
    }
    """Map the container names of properties sequences to the different
    subpop property names."""

    def __init__(
        self,
        individual_classes: Sequence[Type[Individual]],
        species: Sequence[Species],
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

        Each species is evolved in a different subpopulation.

        :param individual_classes: The individual class for each species.
        :type individual_classes: :py:class:`~collections.abc.Sequence` of
            :py:class:`~base.Individual` subclasses
        :param species: The species to be evolved
        :type species: :py:class:`~collections.abc.Sequence` of
            :py:class:`~base.Species`
        :param fitness_function: The training fitness function
        :type fitness_function: :py:class:`~base.FitnessFunction`
        :param subpop_wrapper_cls: Single-population wrapper class to handle
            the subpopulations.
        :type subpop_wrapper_cls: Any subclass of
            :py:class:`~wrapper.single_pop.SinglePop`
        :param num_gens: The number of generations. If set to
            :py:data:`None`, :py:attr:`~wrapper.DEFAULT_NUM_GENS` will
            be used. Defaults to :py:data:`None`
        :type num_gens: :py:class:`int`, optional
        :param pop_sizes: The population size for each subpopulation (species).
            If only a single value is provided, the same size will be used for
            all the subpopulations. Different sizes can be provided in a
            :py:class:`~collections.abc.Sequence`. All the sizes must be
            greater then zero. If set to :py:data:`None`,
            :py:attr:`~wrapper.single_pop.DEFAULT_POP_SIZE` will be used.
            Defaults to :py:data:`None`
        :type pop_sizes: :py:class:`int` or
            :py:class:`~collections.abc.Sequence` of :py:class:`int`, optional
        :param crossover_funcs: The crossover function for each subpopulation
            (species). If only a single value is provided, the same function
            will be used for all the subpopulations. Different functions can
            be provided in a :py:class:`~collections.abc.Sequence`. If set to
            :py:data:`None`, the *individual_cls*
            :py:meth:`~base.Individual.crossover` method will be used.
            Defaults to :py:data:`None`
        :type crossover_funcs: :py:class:`~collections.abc.Callable` or
            :py:class:`~collections.abc.Sequence` of
            :py:class:`~collections.abc.Callable`, optional
        :param mutation_funcs: The mutation function for each subpopulation
            (species). If only a single value is provided, the same function
            will be used for all the subpopulations. Different functions can
            be provided in a :py:class:`~collections.abc.Sequence`. If set to
            :py:data:`None`, the *individual_cls*
            :py:meth:`~base.Individual.mutate` method will be used.
            Defaults to :py:data:`None`
        :type mutation_funcs: :py:class:`~collections.abc.Callable` or
            :py:class:`~collections.abc.Sequence` of
            :py:class:`~collections.abc.Callable`, optional
        :param selection_funcs: The selection function for each subpopulation
            (species). If only a single value is provided, the same function
            will be used for all the subpopulations. Different functions can
            be provided in a :py:class:`~collections.abc.Sequence`. If set to
            :py:data:`None`,
            :py:attr:`~wrapper.single_pop.DEFAULT_SELECTION_FUNC` will be
            used. Defaults to :py:data:`None`
        :type selection_funcs: :py:class:`~collections.abc.Callable` or
            :py:class:`~collections.abc.Sequence` of
            :py:class:`~collections.abc.Callable`, optional
        :param crossover_probs: The crossover probability for each
            subpopulation (species). If only a single value is provided, the
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
            (species). If only a single value is provided, the same probability
            will be used for all the subpopulations. Different probabilities
            can be provided in a :py:class:`~collections.abc.Sequence`. All
            the probabilities must be in (0, 1). If set to :py:data:`None`,
            :py:attr:`~wrapper.single_pop.DEFAULT_MUTATION_PROB` will be used.
            Defaults to :py:data:`None`
        :type mutation_probs: :py:class:`float` or
            :py:class:`~collections.abc.Sequence` of :py:class:`float`,
            optional
        :param gene_ind_mutation_probs: The gene independent mutation
            probability for each subpopulation (species). If only a single
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
            function of each subpopulation (species). If only a single value is
            provided, the same probability will be used for all the
            subpopulations. Different probabilities can be provided in a
            :py:class:`~collections.abc.Sequence`. If set to :py:data:`None`,
            :py:attr:`~wrapper.single_pop.DEFAULT_SELECTION_FUNC_PARAMS` will
            be used. Defaults to :py:data:`None`
        :type selection_funcs_params: :py:class:`dict` or
            :py:class:`~collections.abc.Sequence` of :py:class:`dict`, optional
        :param num_subpops: The number of subpopulations (species). If set to
            :py:data:`None`, the number of species  evolved by the wrapper is
            will be used, otherwise it must match the number of species.
            Defaults to :py:data:`None`
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
            :py:attr:`~wrapper.multi_pop.DEFAULT_REPRESENTATION_TOPOLOGY_FUNC`
            will be used. Defaults to :py:data:`None`
        :type representation_topology_func:
            :py:class:`~collections.abc.Callable`, optional
        :param representation_topology_func_params: Parameters to obtain the
            destinations with the topology function. If set to :py:data:`None`,
            :py:attr:`~wrapper.multi_pop.DEFAULT_REPRESENTATION_TOPOLOGY_FUNC_PARAMS`
            will be used. Defaults to :py:data:`None`
        :type representation_topology_func_params: :py:class:`dict`, optional
        :param representation_selection_func: Policy function to choose the
            representatives from each subpopulation (species). If set to
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
            (species) wrapper
        :type subpop_wrapper_params: keyworded variable-length argument list
        :raises TypeError: If any argument is not of the appropriate type
        :raises ValueError: If any argument has an incorrect value
        """
        self.individual_classes = individual_classes
        self.species = species

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

        # Get the rest of parameters
        self.pop_sizes = pop_sizes
        self.crossover_funcs = crossover_funcs
        self.mutation_funcs = mutation_funcs
        self.selection_funcs = selection_funcs
        self.crossover_probs = crossover_probs
        self.mutation_probs = mutation_probs
        self.gene_ind_mutation_probs = gene_ind_mutation_probs
        self.selection_funcs_params = selection_funcs_params

    @property
    def individual_classes(self) -> Sequence[Type[Individual]]:
        """Get and set the individual classes.

        :getter: Return the current individual classes
        :setter: Set the new individual classes
        :type: :py:class:`~collections.abc.Sequence` of
            :py:class:`~base.Individual` subclasses
        :raises TypeError: If set to a value which is not a
            :py:class:`~collections.abc.Sequence`
        :raises ValueError: If any element of the sequence is not an
            :py:class:`~base.Individual` subclass
        """
        return self._individual_classes

    @individual_classes.setter
    def individual_classes(self, classes: Sequence[Type[Individual]]) -> None:
        """Set the new individual classes.

        :param classes: The classes.
        :type classes: :py:class:`~collections.abc.Sequence` of
            :py:class:`~base.Individual` subclasses
        :raises TypeError: If *classes* is not a
            :py:class:`~collections.abc.Sequence`
        :raises ValueError: If any element of *classes* is not an
            :py:class:`~base.Individual` subclass
        """
        self._individual_classes = check_sequence(
            classes,
            "individual classes",
            item_checker=partial(check_subclass, cls=Individual)
        )

        # Reset the algorithm
        self.reset()

    @property
    def species(self) -> Sequence[Species]:
        """Get and set the species for each subpopulation.

        :getter: Return the current species
        :setter: Set the new species
        :type: :py:class:`~collections.abc.Sequence` of
            :py:class:`~base.Species`
        :raises TypeError: If set to a value which is not a
            :py:class:`~collections.abc.Sequence`
        :raises ValueError: If any element of the sequence is not a
            :py:class:`~base.Species`
        """
        return self._species

    @species.setter
    def species(self, value: Sequence[Species]) -> None:
        """Set the new species.

        :param value: The species.
        :type value: :py:class:`~collections.abc.Sequence` of
            :py:class:`~base.Species`
        :raises TypeError: If *value* is not a
            :py:class:`~collections.abc.Sequence`
        :raises ValueError: If any element of *value* is not a
            :py:class:`~base.Species`
        """
        self._species = check_sequence(
            value,
            "species",
            item_checker=partial(check_instance, cls=Species)
        )

        # Reset the algorithm
        self.reset()

    @property
    def representatives(self) -> Sequence[Sequence[Individual | None]] | None:
        """Return the representatives of all the species."""
        # Default value
        the_representatives = None

        # If the representatives have been gathered before
        if self._representatives is not None:
            the_representatives = self._representatives
        elif self.subpop_wrappers is not None:
            # Create the container for the representatives
            the_representatives = [
                [None] * self.num_subpops
                for _ in range(self.representation_size)
            ]
            for (
                    subpop_index,
                    subpop_wrapper
                    ) in enumerate(self.subpop_wrappers):
                for (context_index,
                     context
                     ) in enumerate(subpop_wrapper.representatives):
                    the_representatives[
                        context_index][
                            subpop_index - 1
                    ] = subpop_wrapper.representatives[
                            context_index][
                                subpop_index - 1
                    ]

            if self._search_finished is True:
                self._representatives = the_representatives

        return the_representatives

    @property
    def num_subpops(self) -> int:
        """Get and set the number of subpopulations.

        :getter: Return the current number of subpopulations
        :setter: Set a new value for the number of subpopulations. If set to
            :py:data:`None`, the number of species  evolved by the wrapper is
            chosen, otherwise it must match the number of species
        :type: :py:class:`int`
        :raises TypeError: If set to a value which is not an integer
        :raises ValueError: If set to a value which is different of the number
            of species
        """
        return (
            len(self.species) if self._num_subpops is None
            else self._num_subpops
        )

    @num_subpops.setter
    def num_subpops(self, value: int | None) -> None:
        """Set the number of subpopulations.

        :param value: The new number of subpopulations. If set to
            :py:data:`None`, the number of species  evolved by the wrapper is
            chosen, otherwise it must match the number of species
        :type value: :py:class:`int`
        :raises TypeError: If *value* is not an integer
        :raises ValueError: If *value* is different of the number of species
        """
        # Check the value
        if value is not None and value != len(self.species):
            raise ValueError(
                "The number of subpopulations must match the number of "
                f"species: {self.species}"
            )

        self._num_subpops = value

        # Reset the algorithm
        self.reset()

    @property
    def representation_topology_func(
        self
    ) -> Callable[[int, int, Any], List[int]]:
        """Get and set the representation topology function.

        :getter: Return the representation topology function
        :setter: Set new representation topology function. If set to
            :py:data:`None`,
            :py:attr:`~wrapper.multi_pop.DEFAULT_REPRESENTATION_TOPOLOGY_FUNC`
            is chosen, otherwise it must match
            :py:attr:`~wrapper.multi_pop.DEFAULT_REPRESENTATION_TOPOLOGY_FUNC`
        :type: :py:class:`~collections.abc.Callable`
        :raises TypeError: If set to a value which is not callable
        :raises ValueError: If set to a value different of
            :py:attr:`~wrapper.multi_pop.DEFAULT_REPRESENTATION_TOPOLOGY_FUNC`
        """
        return (
            DEFAULT_REPRESENTATION_TOPOLOGY_FUNC
            if self._representation_topology_func is None
            else self._representation_topology_func
        )

    @representation_topology_func.setter
    def representation_topology_func(
        self,
        func: Callable[[int, int, Any], List[int]] | None
    ) -> None:
        """Set new representation topology function.

        :param func: The new function. If set to
            :py:data:`None`,
            :py:attr:`~wrapper.multi_pop.DEFAULT_REPRESENTATION_TOPOLOGY_FUNC`
            is chosen, otherwise it must match
            :py:attr:`~wrapper.multi_pop.DEFAULT_REPRESENTATION_TOPOLOGY_FUNC`
        :type func: :py:class:`~collections.abc.Callable`
        :raises TypeError: If *func* is not a callable
        :raises ValueError: If *func* is different of
            :py:attr:`~wrapper.multi_pop.DEFAULT_REPRESENTATION_TOPOLOGY_FUNC`
        """
        # Check func
        if (
            func is not None and
            func != DEFAULT_REPRESENTATION_TOPOLOGY_FUNC
        ):
            raise ValueError(
                "The representation topology function must be "
                f"{DEFAULT_REPRESENTATION_TOPOLOGY_FUNC.__name__}"
            )

        self._representation_topology_func = func

        # Reset the algorithm
        self.reset()

    @property
    def representation_topology_func_params(self) -> Dict[str, Any]:
        """Get and set the parameters of the representation topology function.

        :getter: Return the current parameters for the representation topology
            function
        :setter: Set new parameters. If set to :py:data:`None`,
            :py:attr:`~wrapper.multi_pop.DEFAULT_REPRESENTATION_TOPOLOGY_FUNC_PARAMS`
            is chosen, otherwise it must match
            :py:attr:`~wrapper.multi_pop.DEFAULT_REPRESENTATION_TOPOLOGY_FUNC_PARAMS`
        :type: :py:class:`dict`
        :raises TypeError: If set to a value which is not a :py:class:`dict`
        :raises ValueError: If set to a value different of
            :py:attr:`~wrapper.multi_pop.DEFAULT_REPRESENTATION_TOPOLOGY_FUNC_PARAMS`
        """
        return (
            DEFAULT_REPRESENTATION_TOPOLOGY_FUNC_PARAMS
            if self._representation_topology_func_params is None
            else self._representation_topology_func_params
        )

    @representation_topology_func_params.setter
    def representation_topology_func_params(
        self, params: Dict[str, Any] | None
    ) -> None:
        """Set the parameters for the representation topology function.

        :param params: The new parameters. If set to :py:data:`None`,
            :py:attr:`~wrapper.multi_pop.DEFAULT_REPRESENTATION_TOPOLOGY_FUNC_PARAMS`
            is chosen, otherwise it must match
            :py:attr:`~wrapper.multi_pop.DEFAULT_REPRESENTATION_TOPOLOGY_FUNC_PARAMS`
        :type params: :py:class:`dict`
        :raises TypeError: If *params* is not a :py:class:`dict`
        :raises ValueError: If *params* is different of
            :py:attr:`~wrapper.multi_pop.DEFAULT_REPRESENTATION_TOPOLOGY_FUNC_PARAMS`
        """
        # Check params
        if (
            params is not None and
            params != DEFAULT_REPRESENTATION_TOPOLOGY_FUNC_PARAMS
        ):
            raise ValueError(
                "The representation topology function parameters must be "
                f"{DEFAULT_REPRESENTATION_TOPOLOGY_FUNC_PARAMS}"
            )

        self._representation_topology_func_params = params

        # Reset the algorithm
        self.reset()

    # Copy the :py:class:`culebra.wrapper.multi_pop.HeterogeneousIslands`
    # properties
    pop_sizes = HeterogeneousIslands.pop_sizes
    crossover_funcs = HeterogeneousIslands.crossover_funcs
    mutation_funcs = HeterogeneousIslands.mutation_funcs
    crossover_probs = HeterogeneousIslands.crossover_probs
    mutation_probs = HeterogeneousIslands.mutation_probs
    gene_ind_mutation_probs = HeterogeneousIslands.gene_ind_mutation_probs
    selection_funcs = HeterogeneousIslands.selection_funcs
    selection_funcs_params = HeterogeneousIslands.selection_funcs_params

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
            """Obtain the properties of each subpopulation.

            :raises RuntimeError: If the length of any properties sequence
                does not match the number of subpopulations.

            :return: The properties of each subpopulation.
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

            subpop_wrapper.__class__._init_representatives = partialmethod(
                self._init_subpop_wrapper_representatives,
                individual_classes=self.individual_classes,
                species=self.species,
                representation_size=self.representation_size
            )

            self._subpop_wrappers.append(subpop_wrapper)

    def best_solutions(self) -> Sequence[HallOfFame]:
        """Get the best individuals found for each species.

        :return: A sequence containing :py:class:`~deap.tools.HallOfFame` of
            individuals. One hof for each species
        :rtype: :py:class:`~collections.abc.Sequence`
        """
        # List of hofs
        hofs = []
        # If the search hasn't been initialized a list of empty hofs is
        # returned, one hof per species
        if self.subpop_wrappers is None:
            for _ in range(self.num_subpops):
                hofs.append(HallOfFame(maxsize=None))
        # Else, the best solutions of each species are returned
        else:
            for subpop_wrapper in self.subpop_wrappers:
                hofs.append(subpop_wrapper.best_solutions()[0])

        return hofs

    def best_representatives(self) -> List[List[Individual]] | None:
        """Return a list of representatives from each species.

        :return: A list of representatives lists. One representatives list for
            each one of the evolved species or :py:data:`None` if the search
            has nos finished
        :rtype: :py:class:`list` of :py:class:`list` of
            :py:class:`~base.Individual` or :py:data:`None`
        """
        # Check if the trianing has finished
        # self._search_finished could be None or False...
        if self._search_finished is not True:
            the_representatives = None
        else:
            # Create the container for the representatives
            the_representatives = [
                [None] * self.num_subpops
                for _ in range(self.representation_size)
            ]

            # Get the best solutions for each species
            best_ones = self.best_solutions()

            # Select the representatives
            for species_index, hof in enumerate(best_ones):
                species_representatives = self.representation_selection_func(
                    hof,
                    self.representation_size,
                    **self.representation_selection_func_params
                )

                # Insert the representsatives
                for ind_index, ind in enumerate(species_representatives):
                    the_representatives[ind_index][species_index] = ind

        return the_representatives

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

        anything_received = False
        while not queue.empty():
            msg = queue.get()
            sender_index = msg[0]
            representatives = msg[1]
            for ind_index, ind in enumerate(representatives):
                subpop_wrapper.representatives[ind_index][sender_index] = ind

            anything_received = True

        # If any new representatives have arrived, the fitness of all the
        # individuals in the population must be invalidated and individuals
        # must be re-evaluated
        if anything_received:
            for ind in subpop_wrapper.pop:
                ind.fitness.delValues()
            subpop_wrapper._evaluate_pop(subpop_wrapper.pop)

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

                # Send the following msg:
                # (index of sender subpop, representatives)
                container._communication_queues[dest].put(
                    (subpop_wrapper.index, inds)
                )

    @staticmethod
    def _init_subpop_wrapper_representatives(
        subpop_wrapper,
        individual_classes,
        species,
        representation_size
    ):
        """Init the representatives of the other species.

        This method is used to override dynamically the
        :py:meth:`~base.Wrapper._init_representatives` of all the subpopulation
        wrappers, when they are generated with the
        :py:meth:`~wrapper.multi_pop.Cooperative._generate_subpop_wrappers`
        method, to let them initialize the list of representative individuals
        of the other species

        :param subpop_wrapper: The subpopulation wrapper. The representatives
            from the remaining subpopulation wrappers will be initialized for
            this subpopulation wrapper
        :type subpop_wrapper: :py:class:`~wrapper.single_pop.SinglePop`
        :param individual_classes: The individual class for each species.
        :type individual_classes: :py:class:`~collections.abc.Sequence`
            of :py:class:`~base.Individual` subclasses
        :param species: The species to be evolved by this wrapper
        :type species: :py:class:`~collections.abc.Sequence` of
            :py:class:`~base.Species`
        :param representation_size: Number of representative individuals
            generated for each species
        :type representation_size: :py:class:`int`
        """
        subpop_wrapper._representatives = []

        for _ in range(representation_size):
            subpop_wrapper._representatives.append(
                [
                    ind_cls(
                        spe, subpop_wrapper.fitness_function.Fitness
                    ) if i != subpop_wrapper.index else None
                    for i, (ind_cls, spe) in enumerate(
                        zip(individual_classes, species)
                    )
                ]
            )

    def __copy__(self) -> Cooperative:
        """Shallow copy the wrapper."""
        cls = self.__class__
        result = cls(
            self.individual_classes,
            self.species,
            self.fitness_function,
            self.subpop_wrapper_cls
        )
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo: dict) -> Cooperative:
        """Deepcopy the wrapper.

        :param memo: Wrapper attributes
        :type memo: :py:class:`dict`
        :return: A deep copy of the wrapper
        :rtype: :py:class:`~base.Wrapper`
        """
        cls = self.__class__
        result = cls(
            self.individual_classes,
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
                    self.individual_classes,
                    self.species,
                    self.fitness_function,
                    self.subpop_wrapper_cls
                ),
                self.__dict__)


class SequentialCooperative(Cooperative, SequentialMultiPop):
    """Sequential implementation of the cooperative evolutionary wrapper."""

    def __init__(
        self,
        individual_classes: Type[Individual] | Sequence[Type[Individual]],
        species: Species | Sequence[Species],
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
        Cooperative.__init__(
            self,
            individual_classes=individual_classes,
            species=species,
            fitness_function=fitness_function,
            subpop_wrapper_cls=subpop_wrapper_cls,
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

    # __init__ shares the documentation with Cooperative.__init__
    __init__.__doc__ = Cooperative.__init__.__doc__


class ParallelCooperative(Cooperative, ParallelMultiPop):
    """Parallel implementation of the cooperative evolutionary wrapper."""

    def __init__(
        self,
        individual_classes: Type[Individual] | Sequence[Type[Individual]],
        species: Species | Sequence[Species],
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
        Cooperative.__init__(
            self,
            individual_classes=individual_classes,
            species=species,
            fitness_function=fitness_function,
            subpop_wrapper_cls=subpop_wrapper_cls,
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

    # __init__ shares the documentation with Cooperative.__init__
    __init__.__doc__ = Cooperative.__init__.__doc__


__all__ = [
    'Cooperative',
    'SequentialCooperative',
    'ParallelCooperative'
]
