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

"""Base classes of culebra."""

from __future__ import annotations
from abc import abstractmethod
from typing import Any, Tuple, List, Optional, Dict, Type
from collections.abc import Sequence
from copy import deepcopy
from functools import partial
from multiprocessing.managers import DictProxy
import random
from sklearn.base import ClassifierMixin
from sklearn.naive_bayes import GaussianNB
import numpy as np
from pandas.io.pickle import read_pickle, to_pickle
from deap.base import Fitness as DeapFitness
from deap.tools import Logbook, Statistics, HallOfFame
from . import (
    Base,
    Dataset,
    check_bool,
    check_str,
    check_int,
    check_float,
    check_instance,
    check_sequence
)


__author__ = "Jesús González"
__copyright__ = "Copyright 2021, EFFICOMP"
__license__ = "GNU GPL-3.0-or-later"
__version__ = "0.1.1"
__maintainer__ = "Jesús González"
__email__ = "jesusgonzalez@ugr.es"
__status__ = "Development"


DEFAULT_STATS_NAMES = ("Iter", "NEvals")
"""Default statistics calculated for each iteration of the wrapper."""

DEFAULT_OBJECTIVE_STATS = {
    "Avg": np.mean,
    "Std": np.std,
    "Min": np.min,
    "Max": np.max,
}
"""Default statistics calculated for each objective within a wrapper."""

DEFAULT_CHECKPOINT_ENABLE = True
"""Default checkpointing enablement for wrappers."""

DEFAULT_CHECKPOINT_FREQ = 10
"""Default checkpointing frequency for wrappers."""

DEFAULT_CHECKPOINT_FILENAME = "checkpoint.gz"
"""Default checkpointing file name for wrappers."""

DEFAULT_VERBOSITY = __debug__
"""Default verbosity for wrappers."""

DEFAULT_INDEX = 0
"""Default wrapper index. Only used within distributed wrappers."""

DEFAULT_CLASSIFIER = GaussianNB
"""Default classifier for fitness functions."""


class Fitness(DeapFitness, Base):
    """Define the base class for the fitness of an individual."""

    weights = None
    """A :py:class:`tuple` containing an integer value for each objective
    being optimized. The weights are used in the fitness comparison. They are
    shared among all fitnesses of the same type. When subclassing
    :py:class:`~base.Fitness`, the weights must be defined as a tuple where
    each element is associated to an objective. A negative weight element
    corresponds to the minimization of the associated objective and positive
    weight to the maximization. This attribute is inherited from
    :py:class:`deap.base.Fitness`.

    .. note::
        If weights is not defined during subclassing, the following error will
        occur at instantiation of a subclass fitness object:

        ``TypeError: Can't instantiate abstract <class Fitness[...]> with
        abstract attribute weights.``
    """

    names = None
    """Names of the objectives.

    If not defined, generic names will be used.
    """

    thresholds = None
    """A tuple with the similarity thresholds defined for all the objectives.
    Fitness objects are compared lexicographically. The comparison applies a
    similarity threshold to assume that two fitness values are similar (if
    their difference is lower than or equal to the similarity threshold).
    If not defined, 0 will be used for each objective.
    """

    def __init__(self, values: Tuple[float, ...] = ()) -> None:
        """Construct a default fitness object.

        :param values: Initial values for the fitness, defaults to ()
        :type values: :py:class:`tuple`, optional
        """
        # Init the superclasses
        super().__init__(values)

        if self.names is None:
            suffix_len = len((self.num_obj-1).__str__())

            self.__class__.names = tuple(
                f"obj_{i:0{suffix_len}d}" for i in range(self.num_obj)
            )

        if not isinstance(self.names, Sequence):
            raise TypeError(
                f"Objective names of {self.__class__} must be a sequence")
        if len(self.names) != self.num_obj:
            raise TypeError(
                f"The number of objective names of {self.__class__} must "
                "match its number of weights"
            )

        if self.thresholds is None:
            self.__class__.thresholds = (0.0,) * self.num_obj

        if not isinstance(self.thresholds, Sequence):
            raise TypeError(
                f"Objective thresholds of {self.__class__} must be a sequence")
        if len(self.thresholds) != self.num_obj:
            raise TypeError(
                f"The number of objective thresholds of {self.__class__} must "
                "match its number of weights"
            )

    @property
    def num_obj(self) -> int:
        """Get the number of objectives.

        :type: :py:class:`int`
        """
        return len(self.weights)

    def dominates(self, other: Fitness, which: slice = slice(None)) -> bool:
        """Check if this fitness dominates another one.

        :param other: The other fitness
        :type other: :py:class:`~base.Fitness`
        :param which: Slice indicating on which objectives the domination is
            tested. The default value is :py:class:`slice` (:py:data:`None`),
            representing every objective
        :type which: :py:class:`slice`
        :return: :py:data:`True` if each objective of this fitness is not
            strictly worse than the corresponding objective of the *other* and
            at least one objective is strictly better
        :rtype: :py:class:`bool`
        """
        not_equal = False

        for self_wval, other_wval, threshold in zip(
            self.wvalues[which], other.wvalues[which], self.thresholds[which]
        ):
            if self_wval - other_wval > threshold:
                not_equal = True
            elif other_wval - self_wval > threshold:
                not_equal = False
                break

        return not_equal

    def __hash__(self) -> int:
        """Return the hash number for this individual."""
        return hash(self.wvalues)

    def __gt__(self, other: Fitness) -> bool:
        """Greater than operator.

        :param other: The other fitness
        :type other: :py:class:`~base.Fitness`
        """
        return not self.__le__(other)

    def __ge__(self, other: Fitness) -> bool:
        """Greater than or equal to operator.

        :param other: The other fitness
        :type other: :py:class:`~base.Fitness`
        """
        return not self.__lt__(other)

    def __le__(self, other: Fitness) -> bool:
        """Less than or equal to operator.

        :param other: The other fitness
        :type other: :py:class:`~base.Fitness`
        """
        le = True

        for self_wval, other_wval, threshold in zip(
            self.wvalues, other.wvalues, self.thresholds
        ):
            if other_wval - self_wval > threshold:
                le = True
                break
            if self_wval - other_wval > threshold:
                le = False
                break

        return le

    def __lt__(self, other: Fitness) -> bool:
        """Less than operator.

        :param other: The other fitness
        :type other: :py:class:`~base.Fitness`
        """
        lt = False

        for self_wval, other_wval, threshold in zip(
            self.wvalues, other.wvalues, self.thresholds
        ):
            if other_wval - self_wval > threshold:
                lt = True
                break
            if self_wval - other_wval > threshold:
                lt = False
                break

        return lt

    def __eq__(self, other: Fitness) -> bool:
        """Equal to operator.

        :param other: The other fitness
        :type other: :py:class:`~base.Fitness`
        """
        eq = True

        for self_wval, other_wval, threshold in zip(
            self.wvalues, other.wvalues, self.thresholds
        ):
            if abs(self_wval - other_wval) > threshold:
                eq = False
                break

        return eq

    def __ne__(self, other: Fitness) -> bool:
        """Not equal to operator.

        :param other: The other fitness
        :type other: :py:class:`~base.Fitness`
        """
        return not self.__eq__(other)


class FitnessFunction(Base):
    """Base fitness function."""

    class Fitness(Fitness):
        """Fitness class.

        Handles the values returned by the
        :py:meth:`~base.FitnessFunction.evaluate` method within an
        :py:class:`~base.Individual`.

        This class must be implemented within all the
        :py:class:`~base.FitnessFunction` subclasses, as a subclass of the
        :py:class:`~base.Fitness` class, to define its three class attributes
        (:py:attr:`~base.Fitness.weights`, :py:attr:`~base.Fitness.names`, and
        :py:attr:`~base.Fitness.thresholds`) according to the fitness function.
        """

    def __init__(
        self,
        training_data: Dataset,
        test_data: Optional[Dataset] = None,
        test_prop: Optional[float] = None,
        classifier: ClassifierMixin = DEFAULT_CLASSIFIER()
    ) -> None:
        """Construct a fitness function.

        If *test_data* are provided, the whole *training_data* are used to
        train the *classifier*. Otherwise, if *test_prop* is provided,
        *training_data* are split (stratified) into training and test data.
        Finally, if both *test_data* and *test_prop* are omitted,
        *training_data* are also used to test.

        :param training_data: The training dataset
        :type training_data: :py:class:`~base.Dataset`
        :param test_data: The test dataset, defaults to :py:data:`None`
        :type test_data: :py:class:`~base.Dataset`, optional
        :param test_prop: A real value in [0, 1] or :py:data:`None`. Defaults
            to :py:data:`None`
        :type test_prop: :py:class:`float`, optional
        :param classifier: The classifier, defaults to
            :py:attr:`~base.DEFAULT_CLASSIFIER`
        :type classifier: :py:class:`~sklearn.base.ClassifierMixin`, optional
        """
        # Init the superclass
        super().__init__()

        # Set the attributes to default values
        self.training_data = training_data
        self.test_data = test_data
        self.test_prop = test_prop
        self.classifier = classifier

    @classmethod
    def set_fitness_thresholds(
        cls, thresholds: float | Sequence[float]
    ) -> None:
        """Set new fitness thresholds.

        Modifies the :py:attr:`~Fitness.thresholds` of the
        :py:attr:`~FitnessFunction.Fitness` objects generated by this fitness
        function.

        :param thresholds: The new thresholds. If only a single value
            is provided, the same threshold will be used for all the
            objectives. Different thresholds can be provided in a
            :py:class:`~collections.abc.Sequence`
        :type: :py:class:`float` or :py:class:`~collections.abc.Sequence` of
            :py:class:`float`
        :raises TypeError: If *thresholds* is not a real number or a
            :py:class:`~collections.abc.Sequence` of real numbers
        :raises ValueError: If any threshold is negative
        """
        if isinstance(thresholds, Sequence):
            cls.Fitness.thresholds = check_sequence(
                thresholds,
                "fitness thresholds",
                size=len(cls.Fitness.weights),
                item_checker=partial(check_float, gt=0)
            )
        else:
            cls.Fitness.thresholds = (
                check_float(thresholds, "fitness thresholds", gt=0),
            ) * len(cls.Fitness.weights)

    @property
    def num_obj(self) -> int:
        """Get the number of objectives.

        :type: :py:class:`int`
        """
        return len(self.Fitness.weights)

    @property
    def training_data(self) -> Dataset:
        """Get and set the training dataset.

        :getter: Return the training dataset
        :setter: Set a new training dataset
        :type: :py:class:`~base.Dataset`
        :raises TypeError: If set to an invalid dataset
        """
        return self._training_data

    @training_data.setter
    def training_data(self, value: Dataset) -> None:
        """Set a new training dataset.

        :param value: A new training dataset
        :type value: :py:class:`~base.Dataset`
        :raises TypeError: If set to an invalid dataset
        """
        self._training_data = check_instance(
            value, "training data", cls=Dataset
        )

    @property
    def test_data(self) -> Dataset | None:
        """Get and set the test dataset.

        :getter: Return the test dataset
        :setter: Set a new test dataset
        :type: :py:class:`~base.Dataset`
        :raises TypeError: If set to an invalid dataset
        """
        return self._test_data

    @test_data.setter
    def test_data(self, value: Dataset | None) -> None:
        """Set a new test dataset.

        :param value: A new test dataset
        :type value: :py:class:`~base.Dataset` or :py:data:`None`
        :raises TypeError: If set to an invalid dataset
        """
        self._test_data = None if value is None else check_instance(
            value, "test data", cls=Dataset
        )

    @property
    def test_prop(self) -> float | None:
        """Get and set the proportion of data used to test.

        :getter: Return the test data proportion
        :setter: Set a new value for the test data porportion. A real value in
            [0, 1] or :py:data:`None` is expected
        :type: :py:class:`float`
        :raises TypeError: If set to a value which is not a real number
        :raises ValueError: If set to a value which is not in [0, 1]
        """
        return self._test_prop

    @test_prop.setter
    def test_prop(self, value: float | None) -> None:
        """Set a value for proportion of data used to test.

        :param value: A real value in [0, 1] or :py:data:`None`
        :type value: A real number or :py:data:`None`
        :raises TypeError: If *value* is not a real number
        :raises ValueError: If *value* is not in [0, 1]
        """
        self._test_prop = None if value is None else check_float(
            value, "test proportion", gt=0, lt=1
        )

    @property
    def classifier(self) -> ClassifierMixin:
        """Get and set the classifier applied within this fitness function.

        :getter: Return the classifier
        :setter: Set a new classifier
        :type: :py:class:`~sklearn.base.ClassifierMixin`
        :raises TypeError: If set to a value which is not a classifier
        """
        return self._classifier

    @classifier.setter
    def classifier(self, value: ClassifierMixin) -> None:
        """Set a classifier.

        :param value: The classifier
        :type value: Any subclass of :py:class:`~sklearn.base.ClassifierMixin`
        :raises TypeError: If *value* is not a classifier
        """
        self._classifier = check_instance(
            value, "classifier", cls=ClassifierMixin
        )

    def _final_training_test_data(self) -> Tuple[Dataset, Dataset]:
        """Get the final training and test data.

        If *test_data* is not :py:data:`None`, the whole *training_data* are
        used to train the *classifier*. Otherwise, if *test_prop* is not
        :py:data:`None`, *training_data* are split (stratified) into training
        and test data. Finally, if both *test_data* and *test_prop* are
        :py:data:`None`, *training_data* are also used to test.

        :return: The final training and test datasets
        :rtype: :py:class:`tuple` of :py:class:`~base.Dataset`
        """
        training_data = self.training_data
        test_data = self.test_data

        if test_data is None:
            if self.test_prop is not None:
                # Split training data into training and test
                training_data, test_data = self.training_data.split(
                    self.test_prop
                )
            else:
                # Use the training data also to test
                training_data = test_data = self.training_data

        return training_data, test_data

    @abstractmethod
    def evaluate(
        self,
        ind: Individual,
        index: Optional[int] = None,
        representatives: Optional[Sequence[Individual]] = None
    ) -> Tuple[float, ...]:
        """Evaluate an individual.

        Parameters *representatives* and *index* are used only for cooperative
        evaluations

        This method must be overriden by subclasses to return a correct
        value.

        :param ind: Individual to be evaluated.
        :type ind: :py:class:`~base.Individual`
        :param index: Index where *ind* should be inserted in the
            representatives sequence to form a complete solution for the
            problem
        :type index: :py:class:`int`, optional
        :param representatives: Representative individuals of each species
            being optimized
        :type representatives: A :py:class:`~collections.abc.Sequence`
            containing instances of :py:class:`~base.Individual`,
            optional
        :raises NotImplementedError: if has not been overriden
        :return: The fitness values for *ind*
        :rtype: :py:class:`tuple` of :py:class:`float`
        """
        raise NotImplementedError(
            "The evaluate method has not been implemented in the "
            f"{self.__class__.__name__} class")

    def __copy__(self) -> FitnessFunction:
        """Shallow copy the fitness function."""
        cls = self.__class__
        result = cls(self.training_data)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo: dict) -> FitnessFunction:
        """Deepcopy the fitness function.

        :param memo: Fitness function attributes
        :type memo: :py:class:`dict`
        :return: A deep copy of the fitness function
        :rtype: :py:class:`~base.FitnessFunction`
        """
        cls = self.__class__
        result = cls(self.training_data)
        result.__dict__.update(deepcopy(self.__dict__, memo))
        return result

    def __reduce__(self) -> tuple:
        """Reduce the fitness function.

        :return: The reduction
        :rtype: :py:class:`tuple`
        """
        return (self.__class__, (self.training_data,), self.__dict__)


class Species(Base):
    """Base class for all the species in culebra.

    Each individual evolved within a wrapper must belong to a species which
    constraints the individual parameter values.
    """

    @abstractmethod
    def is_member(self, ind: Individual) -> bool:
        """Check if an individual meets the constraints imposed by the species.

        This method must be overriden by subclasses to return a correct
        value.

        :param ind: The individual
        :type ind: :py:class:`~base.Individual`
        :return: :py:data:`True` if the individual belongs to the species, or
            :py:data:`False` otherwise
        :rtype: :py:class:`bool`
        """
        raise NotImplementedError(
            "The check method has not been implemented in the "
            f"{self.__class__.__name__} class"
        )


class Individual(Base):
    """Base class for all the individuals.

    All the individuals have the following attributes:

    * :py:attr:`~base.Individual.species`: A species with the constraints that
      the individual must meet.
    * :py:attr:`~base.Individual.fitness`: A :py:class:`~base.Fitness` class
      for the individual.
    """

    species_cls = Species
    """Class for the species used by the :py:class:`~base.Individual` class
    to constrain all its instances."""

    def __init__(
        self,
        species: Species,
        fitness_cls: Type[Fitness]
    ) -> None:
        """Construct a default individual.

        :param species: The species the individual will belong to
        :type species: :py:class:`~base.Individual.species_cls`
        :param fitness: The individual's fitness class
        :type fitness: Any subclass of :py:class:`~base.Fitness`
        :raises TypeError: If *species* is not a valid species
        :raises TypeError: If *fitness_cls* is not a valid fitness class
        """
        # Init the superclass
        super().__init__()

        self._species = check_instance(
            species, "species", cls=self.species_cls
        )

        self.fitness = fitness_cls()

    @property
    def species(self) -> Species:
        """Get the individual's species.

        :return: The species
        :rtype: :py:class:`~base.Species`
        """
        return self._species

    @property
    def fitness(self) -> Fitness:
        """Get and set the individual's fitness.

        :getter: Return the current fitness
        :setter: Set a new Fitness
        :type: :py:class:`~base.Fitness`
        """
        return self._fitness

    @fitness.setter
    def fitness(self, value: Fitness) -> None:
        """Set a new fitness for the individual.

        :param value: The new fitness
        :type value: :py:class:`~base.Fitness`
        """
        self._fitness = check_instance(value, "fitness class", cls=Fitness)

    def delete_fitness(self) -> None:
        """Delete the individual's fitness."""
        self._fitness = self.fitness.__class__()

    @abstractmethod
    def crossover(self, other: Individual) -> Tuple[Individual, Individual]:
        """Cross this individual with another one.

        This method must be overriden by subclasses to return a correct
        value.

        :param other: The other individual
        :type other: :py:class:`~base.Individual`
        :raises NotImplementedError: if has not been overriden
        :return: The two offspring
        :rtype: :py:class:`tuple`
        """
        raise NotImplementedError(
            "The crossover operator has not been implemented in the "
            f"{self.__class__.__name__} class"
        )

    @abstractmethod
    def mutate(self, indpb: float) -> Tuple[Individual]:
        """Mutate the individual.

        This method must be overriden by subclasses to return a correct
        value.

        :param indpb: Independent probability for each feature to be mutated.
        :type indpb: :py:class:`float`
        :raises NotImplementedError: if has not been overriden
        :return: The mutant
        :rtype: :py:class:`tuple`
        """
        raise NotImplementedError(
            "The mutation operator has not been implemented in the "
            f"{self.__class__.__name__} class"
        )

    def __hash__(self) -> int:
        """Return the hash number for this individual."""
        return self.__str__().__hash__()

    def dominates(self, other: Individual) -> bool:
        """Dominate operator.

        :param other: Other individual
        :type other: :py:class:`~base.Individual`
        :return: :py:data:`True` if each objective of the individual is not
            strictly worse than the corresponding objective of *other* and at
            least one objective is strictly better.
        :rtype: :py:class:`bool`
        """
        return self.fitness.dominates(other.fitness)

    def __eq__(self, other: Individual) -> bool:
        """Equality test.

        :param other: Other individual
        :type other: :py:class:`~base.Individual`
        :return: :py:data:`True` if the individual codes the same features, or
            :py:data:`False` otherwise
        :rtype: :py:class:`bool`
        """
        return self.__hash__() == other.__hash__()

    def __ne__(self, other: Individual) -> bool:
        """Not equality test.

        :param other: Other individual
        :type other: :py:class:`~base.Individual`
        :return: :py:data:`False` if the individual codes the same features, or
            :py:data:`True` otherwise
        :rtype: :py:class:`bool`
        """
        return not self.__eq__(other)

    def __lt__(self, other: Individual) -> bool:
        """Less than operator.

        :param other: Other individual
        :type other: :py:class:`~base.Individual`
        :return: :py:data:`True` if the individual's fitness is less than the
            *other*'s fitness
        :rtype: :py:class:`bool`
        """
        return self.fitness < other.fitness

    def __gt__(self, other: Individual) -> bool:
        """Greater than operator.

        :param other: Other individual
        :type other: :py:class:`~base.Individual`
        :return: :py:data:`True` if the individual's fitness is greater than
            the *other*'s fitness
        :rtype: :py:class:`bool`
        """
        return self.fitness > other.fitness

    def __le__(self, other: Individual) -> bool:
        """Less than or equal to operator.

        :param other: Other individual
        :type other: :py:class:`~base.Individual`
        :return: :py:data:`True` if the individual's fitness is less than or
            equal to the *other*'s fitness
        :rtype: :py:class:`bool`
        """
        return self.fitness <= other.fitness

    def __ge__(self, other: Individual) -> bool:
        """Greater than or equal to operator.

        :param other: Other individual
        :type other: :py:class:`~base.Individual`
        :return: :py:data:`True` if the individual's fitness is greater than
            or equal to the *other*'s fitness
        :rtype: :py:class:`bool`
        """
        return self.fitness >= other.fitness

    def __copy__(self) -> Individual:
        """Shallow copy the individual."""
        cls = self.__class__
        result = cls(self.species, self.fitness.__class__)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo: dict) -> Individual:
        """Deepcopy the individual.

        :param memo: Individual attributes
        :type memo: :py:class:`dict`
        :return: A deep copy of the individual
        :rtype: :py:class:`~base.Individual`
        """
        cls = self.__class__
        result = cls(self.species, self.fitness.__class__)
        result.__dict__.update(deepcopy(self.__dict__, memo))
        return result

    def __reduce__(self) -> tuple:
        """Reduce the individual.

        :return: The reduction
        :rtype: :py:class:`tuple`
        """
        return (
            self.__class__,
            (self.species, self.fitness.__class__),
            self.__dict__)


class Wrapper(Base):
    """Base class for all the wrapper methods."""

    stats_names = DEFAULT_STATS_NAMES
    """Statistics calculated each iteration."""

    objective_stats = DEFAULT_OBJECTIVE_STATS
    """Statistics calculated for each objective."""

    def __init__(
        self,
        fitness_function: FitnessFunction,
        checkpoint_enable: Optional[bool] = None,
        checkpoint_freq: Optional[int] = None,
        checkpoint_filename: Optional[str] = None,
        verbose: Optional[bool] = None,
        random_seed: Optional[int] = None
    ) -> None:
        """Create a new wrapper.

        :param fitness_function: The training fitness function
        :type fitness_function: :py:class:`~base.FitnessFunction`
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
        # Init the superclass
        super().__init__()

        # Fitness function
        self.fitness_function = fitness_function

        # Configure checkpointing, random seed and verbosity
        self.checkpoint_enable = checkpoint_enable
        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_filename = checkpoint_filename
        self.verbose = verbose
        self.random_seed = random_seed

        # Container wrapper, in case of being used in a distributed
        # configuration
        self.container = None

        # Index of this wrapper, in case of being used in a distributed
        # configuration
        self.index = None

    @staticmethod
    def _get_fitness_values(ind: Individual) -> Tuple[float, ...]:
        """Return the fitness values of an individual.

        DEAP's :py:class:`~deap.tools.Statistics` class needs a function to
        obtain the fitness values of an individual.

        :param ind: The individual
        :type ind: Any subclass of :py:class:`~base.Individual`
        :return: The fitness values of *ind*
        :rtype: :py:class:`tuple`
        """
        return ind.fitness.values

    @property
    def fitness_function(self) -> FitnessFunction:
        """Get and set the training fitness function.

        :getter: Return the fitness function
        :setter: Set a new fitness function
        :type: :py:class:`~base.FitnessFunction`
        :raises TypeError: If set to a value which is not a fitness function
        """
        return self._fitness_function

    @fitness_function.setter
    def fitness_function(self, func: FitnessFunction) -> None:
        """Set a new training fitness function.

        :param func: New training fitness function
        :type func: :py:class:`~base.FitnessFunction`
        :raises TypeError: If set to a value which is not a fitness function
        """
        # Check the function
        self._fitness_function = check_instance(
            func, "fitness_function", FitnessFunction
        )

        # Reset the algorithm
        self.reset()

    @property
    def logbook(self) -> Logbook | None:
        """Get the training logbook.

        Return a logbook with the statistics of the search or :py:data:`None`
        if the search has not been done yet.

        :type: :py:class:`~deap.tools.Logbook`
        """
        return self._logbook

    @property
    def num_evals(self) -> int | None:
        """Get the number of evaluations performed while training.

        Return the number of evaluations or :py:data:`None` if the search has
        not been done yet.

        :type: :py:class:`int`
        """
        return self._num_evals

    @property
    def runtime(self) -> float | None:
        """Get the training runtime.

        Return the training runtime or :py:data:`None` if the search has not
        been done yet.

        :type: :py:class:`float`
        """
        return self._runtime

    @property
    def checkpoint_enable(self) -> bool:
        """Enable or disable checkpointing.

        :getter: Return :py:data:`True` if checkpoinitng is enabled, or
            :py:data:`False` otherwise
        :setter: New value for the checkpoint enablement. If set to
            :py:data:`None`, :py:attr:`~base.DEFAULT_CHECKPOINT_ENABLE` is
            chosen
        :type: :py:class:`bool`
        :raises TypeError: If set to a value which is not boolean
        """
        return (
            DEFAULT_CHECKPOINT_ENABLE if self._checkpoint_enable is None
            else self._checkpoint_enable
        )

    @checkpoint_enable.setter
    def checkpoint_enable(self, value: bool | None) -> None:
        """Enable or disable checkpointing.

        :param value: New value for the checkpoint enablement. If set to
            :py:data:`None`, :py:attr:`~base.DEFAULT_CHECKPOINT_ENABLE` is
            chosen
        :type value: :py:class:`bool`
        :raises TypeError: If *value* is not a boolean value
        """
        self._checkpoint_enable = (
            None if value is None else check_bool(
                value, "checkpoint enablement"
            )
        )

    @property
    def checkpoint_freq(self) -> int:
        """Get and set the checkpoint frequency.

        :getter: Return the checkpoint frequency
        :setter: Set a value for the checkpoint frequency. If set to
            :py:data:`None`, :py:attr:`~base.DEFAULT_CHECKPOINT_FREQ` is chosen
        :type: :py:class:`int`
        :raises TypeError: If set to a value which is not an integer
        :raises ValueError: If set to a value which is not a positive number
        """
        return (
            DEFAULT_CHECKPOINT_FREQ if self._checkpoint_freq is None
            else self._checkpoint_freq
        )

    @checkpoint_freq.setter
    def checkpoint_freq(self, value: int | None) -> None:
        """Set a value for the checkpoint frequency.

        :param value: New value for the checkpoint frequency. If set to
            :py:data:`None`, :py:attr:`~base.DEFAULT_CHECKPOINT_FREQ` is chosen
        :type value: :py:class:`int`
        :raises TypeError: If *value* is not an integer
        :raises ValueError: If *value* is not a positive number
        """
        # Check the value
        self._checkpoint_freq = (
            None if value is None else check_int(
                value, "checkpoint frequency", gt=0
            )
        )

    @property
    def checkpoint_filename(self) -> str:
        """Get and set the checkpoint file path.

        :getter: Return the checkpoint file path
        :setter: Set a new value for the checkpoint file path. If set to
            :py:data:`None`, :py:attr:`~base.DEFAULT_CHECKPOINT_FILENAME` is
            chosen
        :type: :py:class:`str`
        :raises TypeError: If set to a value which is not a string
        """
        return (
            DEFAULT_CHECKPOINT_FILENAME if self._checkpoint_filename is None
            else self._checkpoint_filename
        )

    @checkpoint_filename.setter
    def checkpoint_filename(self, value: str | None) -> None:
        """Set a value for the checkpoint file path.

        :param value: New value for the checkpoint file path. If set to
            :py:data:`None`, :py:attr:`~base.DEFAULT_CHECKPOINT_FILENAME` is
            chosen
        :type value: :py:class:`str`
        :raises TypeError: If * value * is not a string
        """
        # Check the value
        self._checkpoint_filename = (
            None if value is None else check_str(
                value, "checkpoint file name"
            )
        )

    @property
    def random_seed(self) -> int:
        """Get and set the initial random seed used by this wrapper.

        :getter: Return the seed
        :setter: Set a new value for the random seed
        :type: :py:class:`int`
        """
        return self._random_seed

    @random_seed.setter
    def random_seed(self, value: int | None) -> None:
        """Set the random seed for this wrapper.

        :param value: Random seed for the random generator
        :type value: :py:class:`int`
        """
        self._random_seed = value
        random.seed(self._random_seed)
        np.random.seed(self._random_seed)

        # Reset the algorithm
        self.reset()

    @property
    def verbose(self) -> bool:
        """Get and set the verbosity of this wrapper.

        :getter: Return the verbosity
        :setter: Set a new value for the verbosity. If set to
            :py:data:`None`, :py:data:`__debug__` is chosen
        :type: :py:class:`bool`
        :raises TypeError: If set to a value which is not boolean
        """
        return (
            DEFAULT_VERBOSITY if self._verbose is None
            else self._verbose
        )

    @verbose.setter
    def verbose(self, value: bool) -> None:
        """Set the verbosity of this wrapper.

        :param value: The verbosity. If set to :py:data:`None`,
            :py:data:`__debug__` is chosen
        :type value: :py:class:`bool`
        :raises TypeError: If *value* is not boolean
        """
        self._verbose = (
            None if value is None else check_bool(value, "verbosity")
        )

    def _init_representatives(self) -> None:
        """Init the representatives of the other species.

        Only used for cooperative approaches, which need representatives of
        all the species to form a complete solution for the problem.
        Cooperative subclasses of the :py:class:`~base.Wrapper` class should
        override this method to get the representatives of the other species
        initialized.
        """
        self._representatives = None

    @property
    def representatives(self) -> Sequence[Sequence[Individual | None]] | None:
        """Return the representatives of the other species.

        Only used by cooperative wrappers. If the wrapper does not use
        representatives, :py:data:`None` is returned.
        """
        return self._representatives

    @property
    def index(self) -> int:
        """Get and set the wrapper index.

        The wrapper index is only used by distributed wrappers. For the rest
        of wrappers :py:attr:`~base.DEFAULT_INDEX` is used.

        :getter: Return the wrapper index
        :setter: Set a new value for wrapper index. If set to
            :py:data:`None`, :py:attr:`~base.DEFAULT_INDEX` is chosen
        :type: :py:class:`int`
        :raises TypeError: If set to a value which is not an integer
        :raises ValueError: If set to a value which is a negative number
        """
        return (
            DEFAULT_INDEX if self._index is None
            else self._index
        )

    @index.setter
    def index(self, value: int | None) -> None:
        """Set a value for wrapper index.

        The wrapper index is only used by distributed wrappers. For the rest
        of wrappers :py:attr:`~base.DEFAULT_INDEX` is used.

        :param value: New value for the wrapper index. If set to
            :py:data:`None`, :py:attr:`~base.DEFAULT_INDEX` is chosen
        :type value: :py:class:`int`
        :raises TypeError: If *value* is not an integer
        :raises ValueError: If *value* is a negative number
        """
        # Check the value
        self._index = (
            None if value is None else check_int(value, "index", ge=0)
        )

    @property
    def container(self) -> Wrapper | None:
        """Get and set the container of this wrapper.

        The wrapper container is only used by distributed wrappers. For the
        rest of wrappers defaults to :py:data:`None`.

        :getter: Return the container
        :setter: Set a new value for container of this wrapper
        :type: :py:class:`~base.Wrapper`
        :raises TypeError: If set to a value which is not a valid wrapper
        """
        return self._container

    @container.setter
    def container(self, value: Wrapper | None) -> None:
        """Set a container for this wrapper.

        The wrapper container is only used by distributed wrappers. For the
        rest of wrappers defaults to :py:data:`None`.

        :param value: New value for the container or :py:data:`None`
        :type value: :py:class:`~base.Wrapper`
        :raises TypeError: If *value* is not a valid wrapper
        """
        # Check the value
        self._container = (
            None if value is None else check_instance(
                value, "container", cls=Wrapper
            )
        )

    def _save_state(self) -> None:
        """Save the state at a new checkpoint.

        :raises Exception: If the checkpoint file can't be written
        """
        # Save the state
        to_pickle(self._state, self.checkpoint_filename)

    def _load_state(self) -> None:
        """Load the state of the last checkpoint.

        :raises Exception: If the checkpoint file can't be loaded
        """
        # Load the state
        self._state = read_pickle(self.checkpoint_filename)

    @property
    def _state(self) -> Dict[str, Any]:
        """Get and set the state of this wrapper.

        Default state is a dictionary composed of the values of the
        :py:attr:`~base.Wrapper.logbook`,
        :py:attr:`~base.Wrapper.num_evals`,
        :py:attr:`~base.Wrapper.runtime` and
        :py:attr:`~base.Wrapper.representatives`
        wrapper properties, along with a private boolean attribute that informs
        if the search has finished and also the states of the :py:mod:`random`
        and :py:mod:`numpy.random` modules.

        If subclasses use any more properties to keep their state, i.e. the
        number of executed generations for generational wrappers, the
        :py:attr:`~base.Wrapper._state` getter and setter must be overriden to
        take into account such properties.

        :getter: Return the state
        :setter: Set a new state
        :type: :py:class:`dict`
        """
        # Fill in the dictionary with the wrapper state
        return dict(logbook=self._logbook,
                    num_evals=self._num_evals,
                    runtime=self._runtime,
                    representatives=self._representatives,
                    search_finished=self._search_finished,
                    rnd_state=random.getstate(),
                    np_rnd_state=np.random.get_state())

    @_state.setter
    def _state(self, state: Dict[str, Any]) -> None:
        """Set the state of this wrapper.

        :param state: The last loaded state
        :type state: :py:class:`dict`
        """
        self._logbook = state["logbook"]
        self._num_evals = state["num_evals"]
        self._runtime = state["runtime"]
        self._representatives = state["representatives"]
        self._search_finished = state["search_finished"]
        random.setstate(state["rnd_state"])
        np.random.set_state(state["np_rnd_state"])

    def _new_state(self) -> None:
        """Generate a new wrapper state.

        If subclasses add any new property to keep their
        :py:attr:`~base.Wrapper._state`, this method should be overriden to
        initialize the full state of the wrapper.
        """
        # Create a new logbook
        self._logbook = Logbook()

        # Init the logbook
        self._logbook.header = list(self.stats_names) + \
            (self._stats.fields if self._stats else [])

        # Init the number of evaluations
        self._num_evals = 0

        # Init the computing runtime
        self._runtime = 0

        # The wrapper hasn't trained yet
        self._search_finished = False

        # Init the representatives
        self._init_representatives()

    def _init_state(self) -> None:
        """Init the wrapper state.

        If there is any checkpoint file, the state is initialized from it with
        the :py:meth:`~base.Wrapper._load_state` method. Otherwise a new
        initial state is generated with the :py:meth:`~base.Wrapper._new_state`
        method.
        """
        # Init the wrapper state
        create_new_state = True
        if self.checkpoint_enable:
            # Try to load the state of the last checkpoint
            try:
                self._load_state()
                create_new_state = False
            # If a checkpoint can't be loaded, make initialization
            except Exception:
                pass

        if create_new_state:
            self._new_state()

    def _reset_state(self) -> None:
        """Reset the wrapper state.

        If subclasses overwrite the :py:meth:`~base.Wrapper._new_state` method
        to add any new property to keep their :py:attr:`~base.Wrapper._state`,
        this method should also be overriden to reset the full state of the
        wrapper.
        """
        self._logbook = None
        self._num_evals = None
        self._runtime = None
        self._representatives = None
        self._search_finished = None

    def _init_internals(self) -> None:
        """Set up the wrapper internal data structures to start searching.

        Create all the internal objects, functions and data structures needed
        to run the search process. For the :py:class:`~base.Wrapper` class,
        only a :py:class:`~deap.tools.Statistics` object is created. Subclasses
        which need more objects or data structures should override this method.
        """
        # Initialize statistics object
        self._stats = Statistics(self._get_fitness_values)

        # Configure the stats
        for name, func in self.objective_stats.items():
            self._stats.register(name, func, axis=0)

    def _reset_internals(self) -> None:
        """Reset the internal structures of the wrapper.

        If subclasses overwrite the :py:meth:`~base.Wrapper._init_internals`
        method to add any new internal object, this method should also be
        overriden to reset all the internal objects of the wrapper.
        """
        self._stats = None

    def reset(self) -> None:
        """Reset the wrapper.

        Delete the state of the wrapper (with
        :py:meth:`~base.Wrapper._reset_state`) and also all the internal
        data structures needed to perform the search
        (with :py:meth:`~base.Wrapper._reset_internals`).

        This method should be invoqued each time an hyper parameter is
        modified.
        """
        # Reset the wrapper internals
        self._reset_internals()

        # Reset the wrapper state
        self._reset_state()

    def evaluate(
        self,
        ind: Individual,
        fitness_func: Optional[FitnessFunction] = None,
        index: Optional[int] = None,
        representatives: Optional[Sequence[Sequence[Individual | None]]] = None
    ) -> None:
        """Evaluate one individual.

        Its fitness will be modified according with the fitness function
        results.

        :param ind: The individual
        :type ind: :py:class:`~base.Individual`
        :param fitness_func: The fitness function. If omitted, the default
            training fitness function
            (:py:attr:`base.Wrapper.fitness_function`) is used
        :type fitness_func: :py:class:`~base.FitnessFunction`, optional
        :param index: Index where *ind* should be inserted in the
            *representatives* sequence to form a complete solution for the
            problem. If omitted, :py:attr:`base.Wrapper.index` is used
        :type index: :py:class:`int`, optional
        :param representatives: Sequence of representatives of other species
            or :py:data:`None` (if no representatives are needed to evaluate
            *ind*). If omitted, the current value of
            :py:attr:`base.Wrapper.representatives` is used
        :type representatives: :py:class:`~collections.abc.Sequence`
            of :py:class:`~collections.abc.Sequence` of
            :py:class:`~base.Individual` or :py:data:`None`, optional
        """
        # Select the representatives to be used
        context_seq = (
            self.representatives
            if representatives is None
            else representatives
        )

        # Select the fitness function to be used
        if fitness_func is not None:
            # Select the provided function
            func = fitness_func
            # If other fitness function is used, the individual's fitness
            # must be changed
            ind.fitness = func.Fitness()
        else:
            # Select the training fitness function
            func = self.fitness_function
            # Invalidate the last fitness
            ind.fitness.delValues()

        the_index = index if index is not None else self.index

        # If context is not None -> cooperation
        if context_seq is not None:
            for context in context_seq:
                # Calculate their fitness
                fit_values = func.evaluate(
                    ind,
                    index=the_index,
                    representatives=context
                )

                # If it is the first trial ...
                if ind.fitness.valid is False:
                    ind.fitness.values = fit_values
                else:
                    trialFitness = func.Fitness(fit_values)
                    # If the trial fitness is better
                    # TODO Other criteria should be tested to choose
                    # the better fitness estimation
                    if trialFitness > ind.fitness:
                        ind.fitness.values = fit_values
        else:
            ind.fitness.values = func.evaluate(ind, index=the_index)

    @abstractmethod
    def best_solutions(self) -> Sequence[HallOfFame]:
        """Get the best individuals found for each species.

        This method must be overriden by subclasses to return a correct
        value.

        :return: A sequence containing :py:class:`~deap.tools.HallOfFame` of
            individuals. One hof for each species
        :rtype: :py:class:`~collections.abc.Sequence`
        :raises NotImplementedError: If has not been overriden
        """
        raise NotImplementedError(
            "The best_solutions method has not been implemented in the "
            f"{self.__class__.__name__} class"
        )

    def best_representatives(self) -> List[List[Individual]] | None:
        """Return a list of representatives from each species.

        Only used for cooperative wrappers.

        :return: A list of representatives lists if the wrapper is
            cooperative or :py:data:`None` in other cases.
        :rtype: :py:class:`list` of :py:class:`list` of
            :py:class:`~base.Individual` or :py:data:`None`
        """
        return None

    def _init_search(self) -> None:
        """Init the search process.

        Initialize the state of the wrapper (with
        :py:meth:`~base.Wrapper._init_state`) and all the internal data
        structures needed (with :py:meth:`~base.Wrapper._init_internals`) to
        perform the search.
        """
        # Init the wrapper internals
        self._init_internals()

        # Init the state of the wrapper
        self._init_state()

    @abstractmethod
    def _search(self) -> None:
        """Apply the search algorithm.

        This method must be overriden by subclasses to implement the search
        algorithm run by the wrapper.

        :raises NotImplementedError: If has not been overriden
        """
        raise NotImplementedError(
            "The _search method has not been implemented in the "
            f"{self.__class__.__name__} class"
        )

    def _finish_search(self) -> None:
        """Finish the search process.

        This method is called after the search has finished. It can be
        overriden to perform any treatment of the solutions found.
        """
        self._search_finished = True

        # Save the last state
        if self.checkpoint_enable:
            self._save_state()

    def train(self, state_proxy: Optional[DictProxy] = None) -> None:
        """Perform the feature selection process.

        :param state_proxy: Dictionary proxy to copy the output state of the
            wrapper procedure. Only used if train is executed within a
            :py:class:`multiprocessing.Process`. Defaults to :py:data:`None`
        :type state_proxy: :py:class:`~multiprocessing.managers.DictProxy`,
            optional
        """
        # Check state_proxy
        if state_proxy is not None:
            check_instance(state_proxy, "state proxy", cls=DictProxy)

        # Init the search process
        self._init_search()

        # Search the best solutions
        self._search()

        # Finish the search process
        self._finish_search()

        # Copy the ouput state (if needed)
        if state_proxy is not None:
            state = self._state
            for key in state:
                # state_proxy[key] = deepcopy(state[key])
                state_proxy[key] = state[key]

    def test(
        self,
        best_found: Sequence[HallOfFame],
        fitness_func: Optional[FitnessFunction] = None,
        representatives: Optional[Sequence[Sequence[Individual]]] = None
    ) -> None:
        """Apply the test fitness function to the solutions found.

        Update the solutions in *best_found* with their test fitness.

        :param best_found: The best individuals found for each species.
            One :py:class:`~deap.tools.HallOfFame` for each species
        :type best_found: :py:class:`~collections.abc.Sequence` of
            :py:class:`~deap.tools.HallOfFame`
        :param fitness_func: Fitness function used to evaluate the final
            solutions. If ommited, the default training fitness function
            (:py:attr:`base.Wrapper.fitness_function`) will be used
        :type fitness_func: :py:class:`~base.FitnessFunction`, optional
        :param representatives: Sequence of representatives of other species
            or :py:data:`None` (if no representatives are needed). If omitted,
            the current value of :py:attr:`base.Wrapper.representatives` is
            used
        :type representatives: :py:class:`~collections.abc.Sequence` of
            :py:class:`~collections.abc.Sequence` of
            :py:class:`~base.Individual`
        :raises TypeError: If any parameter has a wrong type
        :raises ValueError: If any parameter has an invalid value.
        """
        # Check best_found
        check_sequence(
            best_found,
            "best found individuals",
            item_checker=partial(check_instance, cls=HallOfFame)
        )

        # Check fitness_function, if provided
        if fitness_func is not None:
            check_instance(
                fitness_func, "fitness function", cls=FitnessFunction
            )

        # Check representatives, if provided
        if representatives is not None:
            check_sequence(
                representatives,
                "representatives",
                item_checker=partial(check_instance, cls=Sequence)
            )
            for context in representatives:
                check_sequence(
                    context,
                    "representatives",
                    size=len(best_found),
                    item_checker=partial(check_instance, cls=Individual)
                )

        # For each hof
        for species_index, hof in enumerate(best_found):
            # For each solution found in this hof
            for ind in hof:
                self.evaluate(
                    ind, fitness_func, species_index, representatives
                )

    def __copy__(self) -> Wrapper:
        """Shallow copy the wrapper."""
        cls = self.__class__
        result = cls(self.fitness_function)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo: dict) -> Wrapper:
        """Deepcopy the wrapper.

        :param memo: Wrapper attributes
        :type memo: :py:class:`dict`
        :return: A deep copy of the wrapper
        :rtype: :py:class:`~base.Wrapper`
        """
        cls = self.__class__
        result = cls(self.fitness_function)
        result.__dict__.update(deepcopy(self.__dict__, memo))
        return result

    def __reduce__(self) -> tuple:
        """Reduce the wrapper.

        :return: The reduction
        :rtype: :py:class:`tuple`
        """
        return (self.__class__, (self.fitness_function,), self.__dict__)


# Exported symbols for this module
__all__ = [
    'Fitness',
    'FitnessFunction',
    'Species',
    'Individual',
    'Wrapper',
    'DEFAULT_STATS_NAMES',
    'DEFAULT_OBJECTIVE_STATS',
    'DEFAULT_CHECKPOINT_ENABLE',
    'DEFAULT_CHECKPOINT_FREQ',
    'DEFAULT_CHECKPOINT_FILENAME',
    'DEFAULT_VERBOSITY',
    'DEFAULT_INDEX',
    'DEFAULT_CLASSIFIER'
]
