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

"""Provides the :py:class:`~base.individual.Individual` class."""


import copy
import numpy as np
from culebra.base.species import Species
from culebra.base.fitness import Fitness

__author__ = 'Jesús González'
__copyright__ = 'Copyright 2021, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.1.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


class Individual:
    """Base class for all the individuals."""

    def __init__(self, species, fitness, features=None):
        """Create an individual.

        :param species: The species the individual will belong to
        :type species: :py:class:`~base.species.Species`
        :param fitness: The fitness object for the individual
        :type fitness: Instance of any subclass of
            :py:class:`~base.fitness.Fitness`
        :param features: Features to init the individual. If `None` the
            individual is initialized randomly.
        :type features: Any subclass of :py:class:`~collections.abc.Sequence`
            containing valid feature indices
        :raises TypeError: If any parameter type is wrong
        """
        if not isinstance(species, Species):
            raise TypeError("Not valid species")
        if not isinstance(fitness, Fitness):
            raise TypeError("Not valid fitness")

        # Assing the species
        self._species = species

        # Assing an empty fitness
        self.fitness = copy.copy(fitness)

    @property
    def species(self):
        """The individuals species.

        :type: :py:class:`~base.species.Species`
        """
        return self._species

    @property
    def features(self):
        """Indices of the features selected by the individual.

        This property must be overriden by subclasses to return a correct
        value.

        :getter: Return the indices of the selected features
        :setter: Set the new feature indices. An array-like object of
            feature indices is expected

        :raises NotImplementedError: if has not been overriden
        :type: Array-like object
        """
        raise NotImplementedError("The features property has not been "
                                  "implemented in the "
                                  f"{self.__class__.__name__} class")

    @features.setter
    def features(self, values):
        """Set the indices of the new features selected by the individual.

        This property setter must be overriden by subclasses.

        :param values: The new feature indices
        :type values: Array-like object
        :raises NotImplementedError: if has not been overriden
        """
        raise NotImplementedError("The features property seter has not been "
                                  "implemented in the "
                                  f"{self.__class__.__name__} class")

    @property
    def num_feats(self):
        """Number of features selected by the individual.

        This property must be overriden by subclasses to return a correct
        value.

        :raises NotImplementedError: if has not been overriden
        :type: :py:class:`int`
        """
        raise NotImplementedError("The num_feats property has not been "
                                  "implemented in the "
                                  f"{self.__class__.__name__} class")

    @property
    def min_feat(self):
        """Minimum feature index selected by the individual.

        This property must be overriden by subclasses to return a correct
        value.

        :raises NotImplementedError: if has not been overriden
        :type: :py:class:`int`
        """
        raise NotImplementedError("The min_feat property has not been "
                                  "implemented in the "
                                  f"{self.__class__.__name__} class")

    @property
    def max_feat(self):
        """Maximum feature index selected by the individual.

        This property must be overriden by subclasses to return a correct
        value.

        :raises NotImplementedError: if has not been overriden
        :type: :py:class:`int`
        """
        raise NotImplementedError("The max_feat property has not been "
                                  "implemented in the "
                                  f"{self.__class__.__name__} class")

    def dominates(self, other, which=slice(None)):
        """Dominate operator.

        :param other: Other individual
        :type other: :py:class:`~base.individual.Individual`
        :param which: Slice object indicating on which objectives the
            domination is tested. The default value is `slice(None)`,
            representing every objectives.
        :type which: :py:class:`slice`
        :return: `True` if each objective of *self* is not strictly worse than
            the corresponding objective of *other* and at least one objective
            is strictly better.
        :rtype: :py:class:`bool`
        """
        return self.fitness.dominates(other.fitness, which)

    def crossover(self, other):
        """Cross this individual with another one.

        This method must be overriden by subclasses to return a correct
        value.

        :param other: The other individual
        :type other: :py:class:`~base.individual.Individual`
        :raises NotImplementedError: if has not been overriden
        :return: The two offspring
        :rtype: :py:class:`tuple`
        """
        raise NotImplementedError("The crossover operator has not been "
                                  "implemented in the "
                                  f"{self.__class__.__name__} class")

    def mutate(self, indpb):
        """Mutate the individual.

        This method must be overriden by subclasses to return a correct
        value.

        :param indpb: Independent probability for each feature to be mutated.
        :type indpb: :py:class:`float`
        :raises NotImplementedError: if has not been overriden
        :return: The mutant
        :rtype: :py:class:`tuple`
        """
        raise NotImplementedError("The mutation operator has not been "
                                  "implemented in the "
                                  f"{self.__class__.__name__} class")

    def _check_features(self, values):
        """Check if a set of feature indices meet the species constraints.

        :param values: The new feature indices
        :type values: Array-like object
        :raises ValueError: If the values do not meet the species constraints.
        """
        # Get the set of indices (ordered)
        indices = np.unique(np.asarray(values, dtype=int))
        size = indices.shape[0]

        # Check the species constraints
        if size < self.species.min_size:
            raise ValueError("Too few features. The minimum size is "
                             f"{self.species.min_size}")
        elif size > self.species.max_size:
            raise ValueError("Too many features. The maximum size is "
                             f"{self.species.max_size}")
        elif size > 0:
            min_value = indices[0]
            max_value = indices[-1]
            if min_value < self.species.min_feat:
                raise ValueError(f"{min_value} is lower than the species "
                                 f"min_feat ({self.species.min_feat})")
            elif max_value > self.species.max_feat:
                raise ValueError(f"{max_value} is higher than the species "
                                 f"max_feat ({self.species.max_feat})")

        return indices

    def __str__(self):
        """Return the individual as a string."""
        return self.features.__str__()

    def __repr__(self):
        """Return the individual representation."""
        cls_name = self.__class__.__name__
        species_info = self.species.__str__()
        fitness_info = self.fitness.values

        return (f"{cls_name}(species={species_info}, fitness={fitness_info}, "
                f"features={self.__str__()})")

    def __setstate__(self, state):
        """Set the state of the individual.

        :param state: The state
        :type state: :py:class:`dict`
        """
        self.__dict__.update(state)

    def __reduce__(self):
        """Reduce the individual.

        :return: The reduction
        :rtype: :py:class:`tuple`
        """
        return (self.__class__,
                (self._species, self.fitness),
                self.__dict__)

    def __copy__(self):
        """Shallow copy."""
        cls = self.__class__
        result = cls(self.species, self.fitness)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        """Deepcopy the individual.

        :param memo: Individual attributes
        :type memo: :py:class:`dict`
        :return: A deep copy of the individual
        :rtype: :py:class:`~base.individual.Individual`
        """
        cls = self.__class__
        result = cls(self.species, self.fitness)
        result.__dict__.update(copy.deepcopy(self.__dict__, memo))
        return result

    def __hash__(self):
        """Return the hash number for this individual."""
        return self.__str__().__hash__()

    def __eq__(self, other):
        """Equality test.

        :param other: Other individual
        :type other: :py:class:`~base.individual.Individual`
        :return: `True` if the individual codes the same features. `False`
            otherwise
        :rtype: :py:class:`bool`
        """
        return self.__hash__() == other.__hash__()

    def __ne__(self, other):
        """Not equality test.

        :param other: Other individual
        :type other: :py:class:`~base.individual.Individual`
        :return: `False` if the individual codes the same features. `True`
            otherwise
        :rtype: :py:class:`bool`
        """
        return not self.__eq__(other)

    def __lt__(self, other):
        """Less than operator.

        :param other: Other individual
        :type other: :py:class:`~base.individual.Individual`
        :return: `True` if the individual's fitness is less than the other's
            fitness
        :rtype: :py:class:`bool`
        """
        return self.fitness < other.fitness

    def __gt__(self, other):
        """Greater than operator.

        :param other: Other individual
        :type other: :py:class:`~base.individual.Individual`
        :return: `True` if the individual's fitness is greater than the other's
            fitness
        :rtype: :py:class:`bool`
        """
        return self.fitness > other.fitness

    def __le__(self, other):
        """Less than or equal to operator.

        :param other: Other individual
        :type other: :py:class:`~base.individual.Individual`
        :return: `True` if the individual's fitness is less than or equal to
            the other's fitness
        :rtype: :py:class:`bool`
        """
        return self.fitness <= other.fitness

    def __ge__(self, other):
        """Greater than or equal to operator.

        :param other: Other individual
        :type other: :py:class:`~base.individual.Individual`
        :return: `True` if the individual's fitness is greater than or equal to
            the other's fitness
        :rtype: :py:class:`bool`
        """
        return self.fitness >= other.fitness
