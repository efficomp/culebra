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

"""Fitness functions for feature selection problems."""

from __future__ import annotations
from typing import Optional, Tuple
from collections.abc import Sequence
from sklearn.metrics import cohen_kappa_score
from culebra.base import Fitness, FitnessFunction
from culebra.genotype.feature_selection import Individual


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2021, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.1.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


DEFAULT_THRESHOLD = 0.01
"""Default similarity threshold for fitnesses."""


class NumFeats(FitnessFunction):
    """Dummy single-objective fitness function for testing purposes.

    Return the number of selected features by an individual.
    """

    class Fitness(Fitness):
        """Fitness class.

        Handles the values returned by the
        :py:meth:`~fitness_function.feature_selection.NumFeats.evaluate`
        method within an :py:class:`~genotype.feature_selection.Individual`.
        """

        weights = (-1.0,)
        """Minimize the number of features that an individual has selected."""

        names = ("NF",)
        """Name of the objective."""

        thresholds = (DEFAULT_THRESHOLD,)
        """Similarity threshold for fitness comparisons."""

    def evaluate(
        self,
        ind: Individual,
        index: Optional[int] = None,
        representatives: Optional[Sequence[Individual]] = None
    ) -> Tuple[float, ...]:
        """Evaluate an individual.

        :param ind: Individual to be evaluated.
        :type ind: :py:class:`~genotype.feature_selection.Individual`
        :param index: Index where *ind* should be inserted in the
            representatives sequence to form a complete solution for the
            problem. Only used by cooperative problems
        :type index: :py:class:`int`, ignored
        :param representatives: Representative individuals of each species
            being optimized. Only used by cooperative problems
        :type representatives: :py:class:`~collections.abc.Sequence` of
            :py:class:`~base.Individual`, ignored
        :return: The fitness of *ind*
        :rtype: :py:class:`tuple` of :py:class:`float`
        """
        # Return the individual's size
        return (ind.num_feats,)


class KappaIndex(FitnessFunction):
    """Single-objective fitness function for classification problems.

    Calculate the Kohen's Kappa index.
    """

    class Fitness(Fitness):
        """Fitness class.

        Handles the values returned by the
        :py:meth:`~fitness_function.feature_selection.KappaIndex.evaluate`
        method within an :py:class:`~genotype.feature_selection.Individual`.
        """

        weights = (1.0,)
        """Maximizes the valiation Kappa index."""

        names = ("Kappa",)
        """Name of the objective."""

        thresholds = (DEFAULT_THRESHOLD,)
        """Similarity threshold for fitness comparisons."""

    def evaluate(
        self,
        ind: Individual,
        index: Optional[int] = None,
        representatives: Optional[Sequence[Individual]] = None
    ) -> Tuple[float, ...]:
        """Evaluate an individual.

        :param ind: Individual to be evaluated.
        :type ind: :py:class:`~genotype.feature_selection.Individual`
        :param index: Index where *ind* should be inserted in the
            representatives sequence to form a complete solution for the
            problem. Only used by cooperative problems
        :type index: :py:class:`int`, ignored
        :param representatives: Representative individuals of each species
            being optimized. Only used by cooperative problems
        :type representatives: :py:class:`~collections.abc.Sequence` of
            :py:class:`~base.Individual`, ignored
        :return: The fitness of *ind*
        :rtype: :py:class:`tuple` of :py:class:`float`
        """
        kappa = 0

        if ind.features.size > 0:
            # Get the training and test data
            training_data, test_data = self._final_training_test_data()

            # Train and get the outputs for the validation data
            outputs_pred = self.classifier.fit(
                training_data.inputs[:, ind.features],
                training_data.outputs
            ).predict(test_data.inputs[:, ind.features])
            kappa = cohen_kappa_score(test_data.outputs, outputs_pred)

        return (kappa,)


class KappaNumFeats(KappaIndex, NumFeats):
    """Bi-objective fitness class for feature selection.

    Maximizes the Kohen's Kappa index and minimizes the number of features
    that an individual has selected.
    """

    class Fitness(Fitness):
        """Fitness class.

        Handles the values returned by the
        :py:meth:`~fitness_function.feature_selection.KappaNumFeats.evaluate`
        method within an :py:class:`~genotype.feature_selection.Individual`.
        """

        weights = KappaIndex.Fitness.weights + NumFeats.Fitness.weights
        """Maximizes the Kohen's Kappa index and minimizes the number of
        features that an individual has selected.
        """

        names = KappaIndex.Fitness.names + NumFeats.Fitness.names
        """Name of the objectives."""

        thresholds = (
            KappaIndex.Fitness.thresholds + NumFeats.Fitness.thresholds
        )
        """Similarity thresholds for fitness comparisons."""

    def evaluate(
        self,
        ind: Individual,
        index: Optional[int] = None,
        representatives: Optional[Sequence[Individual]] = None
    ) -> Tuple[float, ...]:
        """Evaluate an individual.

        :param ind: Individual to be evaluated.
        :type ind: :py:class:`~genotype.feature_selection.Individual`
        :param index: Index where *ind* should be inserted in the
            representatives sequence to form a complete solution for the
            problem. Only used by cooperative problems
        :type index: :py:class:`int`, ignored
        :param representatives: Representative individuals of each species
            being optimized. Only used by cooperative problems
        :type representatives: :py:class:`~collections.abc.Sequence` of
            :py:class:`~base.Individual`, ignored
        :return: The fitness of *ind*
        :rtype: :py:class:`tuple` of :py:class:`float`
        """
        return KappaIndex.evaluate(self, ind) + NumFeats.evaluate(self, ind)


# Exported symbols for this module
__all__ = ['NumFeats', 'KappaIndex', 'KappaNumFeats']
