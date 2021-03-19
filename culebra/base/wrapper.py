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

"""Provides the :py:class:`~base.wrapper.Wrapper` class."""

import numbers
import random
import numpy as np
from deap.tools import Statistics
from culebra.base.individual import Individual
from culebra.base.species import Species
from culebra.base.fitness import Fitness
from culebra.base.dataset import Dataset

__author__ = 'Jesús González'
__copyright__ = 'Copyright 2021, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.1.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'

DEFAULT_STATS_NAMES = ("N",)
"""Default statistics calculated each time."""

DEFAULT_OBJECTIVE_STATS = {"Avg": np.mean, "Std": np.std, "Min": np.min,
                           "Max": np.max}
"""Default statistics calculated for each objective."""

DEFAULT_CHECKPOINT_FREQ = 10
"""Default frequency for checkpointing."""

DEFAULT_CHECKPOINT_FILE = "checkpoint.gz"
"""Default file for checkpointing."""


class Wrapper:
    """Base class for all the wrapper methods."""

    stats_names = DEFAULT_STATS_NAMES
    """Statistics calculated each time."""

    objective_stats = DEFAULT_OBJECTIVE_STATS
    """Statistics calculated for each objective."""

    def __init__(self, individual_cls, species, **params):
        """Initialize the wrapper method.

        :param individual_cls: Individual representation.
        :type individual_cls: Any subclass of
            :py:class:`~base.individual.Individual`
        :param species: The species the individual will belong to
        :type species: :py:class:`~base.species.Species`
        :param checkpoint_freq: Frequency for checkpointing, defaults to
            :py:attr:`~base.wrapper.DEFAULT_CHECKPOINT_FREQ`
        :type checkpoint_freq: :py:class:`int`, optional
        :param checkpoint_file: File path for checkpointing, defaults to
            :py:attr:`~base.wrapper.DEFAULT_CHECKPOINT_FILE`
        :type checkpoint_file: :py:class:`str`, optional
        :param random_seed: Random seed for the random generator, defaults to
            `None`
        :type random_seed: :py:class:`int`, optional
        :param verbose: Whether or not to log the statistics, defaults to
            :py:data:`__debug__`
        :type verbose: :py:class:`bool`
        :raises TypeError: If any parameter has a wrong type
        """
        # Check the individual class
        if not (isinstance(individual_cls, type) and
                issubclass(individual_cls, Individual)):
            raise TypeError("Not valid individual class")
        self._individual_cls = individual_cls

        # Check the species
        if not isinstance(species, Species):
            raise TypeError("Not valid species")
        self._species = species

        # Get the checkpoint frequency
        self.checkpoint_freq = params.pop('checkpoint_freq',
                                          DEFAULT_CHECKPOINT_FREQ)

        # Get the checkpoint file
        self.checkpoint_file = params.pop('checkpoint_file',
                                          DEFAULT_CHECKPOINT_FILE)

        # Set the random seed for the number generator
        self.random_seed = params.pop('random_seed', None)

        # Set the verbosity of the algorithm
        self.verbose = params.pop('verbose', __debug__)

        # Initialize statistics object
        self._stats = Statistics(self._get_fitness_values)

        # Configure the stats
        for name, func in self.objective_stats.items():
            self._stats.register(name, func, axis=0)

    @staticmethod
    def _get_fitness_values(ind):
        """Return the fitness values of an individual.

        DEAP :py:class:`~deap.tools.Statistics` class needs a function to
        obtain the fitness values of an individual.

        :param ind: The individual
        :type ind: Any subclass of :py:class:`~base.individual.Individual`
        :return: The fitness values of *ind*
        :rtype: :py:class:`tuple`
        """
        return ind.fitness.values

    @property
    def objective_stats_names(self):
        """Names of the objectives stats.

        :type: :py:class:`tuple` of :py:class:`str`
        """
        return tuple(self.objective_stats.keys())

    @property
    def checkpoint_freq(self):
        """Checkpoint frequency.

        :getter: Return the checkpoint frequency
        :setter: Set a value for the checkpoint frequency
        :raises TypeError: If set to a value which is not an integer
        :raises ValueError: If set to a value which is not a positive number
        :type: :py:class:`int`
        """
        return self._checkpoint_freq

    @checkpoint_freq.setter
    def checkpoint_freq(self, value):
        """Set a value for the checkpoint frequency.

        :param value: New value for the checkpoint frequency
        :type value: :py:class:`int`
        :raises TypeError: If *value* is not an integer
        :raises ValueError: If *value* is not a positive number
        """
        # Check the value
        if not isinstance(value, numbers.Integral):
            raise TypeError("The checkpoint frequency must be an integer "
                            "number")
        if value <= 0:
            raise ValueError("The checkpoint frequency must be a positive "
                             "number")

        self._checkpoint_freq = value

    @property
    def checkpoint_file(self):
        """Checkpoint file path.

        :getter: Return the checkpoint file path
        :setter: Set a new value for the checkpoint file path
        :type: :py:class:`str`
        :raises TypeError: If set to a value which is not a string
        """
        return self._checkpoint_file

    @checkpoint_file.setter
    def checkpoint_file(self, value):
        """Set a value for the checkpoint file path.

        :param value: New value for the checkpoint file path
        :type value: :py:class:`str`
        :raises TypeError: If *value* is not a string
        """
        # Check the value
        if not isinstance(value, str):
            raise TypeError("The checkpoint file path must be a string")

        self._checkpoint_file = value

    @property
    def random_seed(self):
        """Initial random seed used by this wrapper.

        :getter: Return the seed
        :setter: Set a new value for the random seed
        :type: :py:class:`int`
        """
        return self._random_seed

    @random_seed.setter
    def random_seed(self, value):
        """Set the random seed for this wrapper.

        :param value: Random seed for the random generator
        :type value: :py:class:`int`
        """
        self._random_seed = value
        random.seed(self._random_seed)
        np.random.seed(self._random_seed)

    @property
    def verbose(self):
        """Verbosity of this wrapper.

        :getter: Return the verbosity
        :setter: Set a new value for the verbosity
        :type: :py:class:`bool`
        :raises TypeError: If set to a value which is not boolean
        """
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        """Set the verbosity of this wrapper.

        :param value: `True` of `False`
        :type value: :py:class:`bool`
        :raises TypeError: If *value* is not boolean
        """
        if not isinstance(value, bool):
            raise TypeError("Verbose must be a boolean value")
        self._verbose = value

    def _search(self):
        """Application of the search algorithm.

        This method must be overriden by subclasses to return a correct
        value.

        :raises NotImplementedError: if has not been overriden
        :return: The best individuals found
        :rtype: :py:class:`~deap.tools.HallOfFame` of individuals
        :return: A logbook with the statistics of the evolution
        :rtype: :py:class:`~deap.tools.Logbook`
        :return: The runtime of the algorithm
        :rtype: :py:class:`float`
        """
        raise NotImplementedError("The search method has not been implemented "
                                  f"in the {self.__class__.__name__} class")

    def __check_dataset_and_fitness(self, dataset, fitness):
        """Check the parameteres before training or testing.

        :param dataset: A dataset
        :type dataset: :py:class:`~base.dataset.Dataset`
        :param fitness: Fitness used while training or testing
        :type fitness: Any subclass of :py:class:`~base.fitness.Fitness`
        :raises TypeError: If any parameter has a wrong type
        :raises RuntimeError: If the number of features in the dataset
            does not match that of the species.
        """
        # Check the fitness class
        if not (isinstance(fitness, Fitness)):
            raise TypeError("Not valid fitness class")

        # Check the dataset
        if not (isinstance(dataset, Dataset)):
            raise TypeError("Not valid dataset")

        # Check if the number of features in the dataset matches that in the
        # species
        if dataset.num_feats != self._species.num_feats:
            raise RuntimeError("The number of features in the dataset "
                               f"({dataset.num_feats}) is different than that "
                               "initialized in the species "
                               f"({self._species.num_feats})")

    def train(self, dataset, fitness):
        """Perform the feature selection process.

        :param dataset: Training dataset
        :type dataset: :py:class:`~base.dataset.Dataset`
        :param fitness: Fitness used while training
        :type fitness: Any subclass of :py:class:`~base.fitness.Fitness`
        :raises TypeError: If any parameter has a wrong type
        :raises RuntimeError: If the number of features in the training data
            does not match that of the species.
        :return: The best individuals found
        :rtype: :py:class:`~deap.tools.HallOfFame` of individuals
        :return: A logbook with the statistics of the evolution
        :rtype: :py:class:`~deap.tools.Logbook`
        :return: The runtime of the algorithm
        :rtype: :py:class:`float`
        """
        # Check the parameters
        self.__check_dataset_and_fitness(dataset, fitness)

        # Register the function to initialize new individuals in the toolbox
        self._toolbox.register("individual", self._individual_cls,
                               self._species, fitness)

        # Register the evaluation function
        self._toolbox.register("evaluate", fitness.eval, dataset=dataset)

        # Search the best solutions
        return self._search()

    def test(self, hof, dataset, fitness):
        """Apply the test data to the solutions found by the wrapper method.

        Update the solutions in *hof* with their test fitness.

        :param hof: The best individuals found
        :type hof: :py:class:`~deap.tools.HallOfFame` of individuals
        :param dataset: Test dataset
        :type dataset: :py:class:`~base.dataset.Dataset`
        :param fitness: Fitness used to evaluate the final solutions
        :type fitness: Any subclass of :py:class:`~base.fitness.Fitness`
        :raises TypeError: If any parameter has a wrong type
        :raises RuntimeError: If the number of features in the test data does
            not match that of the species.
        """
        # Check the parameters
        self.__check_dataset_and_fitness(dataset, fitness)

        # For each solution found
        for ind in hof:
            ind.fitness.setValues(fitness.eval(ind, dataset))

    def __setstate__(self, state):
        """Set the state of the wrapper.

        :param state: The state
        :type state: :py:class:`dict`
        """
        self.__dict__.update(state)

    def __reduce__(self):
        """Reduce the wrapper.

        :return: The reduction
        :rtype: :py:class:`tuple`
        """
        return (self.__class__, (self._individual_cls, self._species),
                self.__dict__)

    def __repr__(self):
        """Return the wrapper representation."""
        cls = self.__class__
        cls_name = cls.__name__
        properties = (
                p for p in dir(cls)
                if isinstance(getattr(cls, p), property) and
                not p.startswith('_')
                      )

        repr = cls_name
        sep = "("
        for p in properties:
            repr += sep + p + "=" + getattr(self, p).__str__()
            sep = ", "

        repr += ")"
        return repr
