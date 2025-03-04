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

"""Feature selection related fitness functions.

This sub-module provides several fitness functions to solve feature selecion
problems:

  * :py:class:`~culebra.fitness_function.feature_selection.NumFeats`: Dummy
    single-objective function that minimizes the number of selected features
    from a :py:class:`~culebra.tools.Dataset`.

  * :py:class:`~culebra.fitness_function.feature_selection.FeatsProportion`:
    Dummy single-objective function that minimizes the number of selected
    features from a :py:class:`~culebra.tools.Dataset`. The difference with
    :py:class:`~culebra.fitness_function.feature_selection.NumFeats` is just
    that
    :py:class:`~culebra.fitness_function.feature_selection.FeatsProportion`
    returns a normalized number in [0, 1].

  * :py:class:`~culebra.fitness_function.feature_selection.KappaIndex`:
    Single-objective function that maximizes the Kohen's Kappa index for
    classification problems.

  * :py:class:`~culebra.fitness_function.feature_selection.Accuracy`:
    Single-objective function that maximizes the Accuracy for classification
    problems.

  * :py:class:`~culebra.fitness_function.feature_selection.KappaNumFeats`:
    Bi-objective function composed by the two former functions. It tries to
    both maximize the Kohen's Kappa index and minimize the number of features
    that a solution has selected.

  * :py:class:`~culebra.fitness_function.feature_selection.AccuracyNumFeats`:
    Bi-objective function composed by the two former functions. It tries to
    both maximize the Accuracy and minimize the number of features that a
    solution has selected.

  * :py:class:`~culebra.fitness_function.feature_selection.KappaFeatsProp`:
    Bi-objective function composed by the two former functions. It tries to
    both maximize the Kohen's Kappa index and minimize the proportion of
    features that a solution has selected.

  * :py:class:`~culebra.fitness_function.feature_selection.AccuracyFeatsProp`:
    Bi-objective function composed by the two former functions. It tries to
    both maximize the Accuracy and minimize the proportion of features that a
    solution has selected.
"""

from __future__ import annotations

from typing import Tuple, Optional
from collections.abc import Sequence

from sklearn.metrics import cohen_kappa_score, accuracy_score

from culebra.abc import Fitness
from culebra.fitness_function import DEFAULT_THRESHOLD
from culebra.fitness_function.abc import FeatureSelectionFitnessFunction
from culebra.solution.feature_selection import Solution


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2023, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.3.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


class NumFeats(FeatureSelectionFitnessFunction):
    """Dummy single-objective fitness function for testing purposes.

    Return the number of selected features by a solution.
    """

    class Fitness(Fitness):
        """Fitness class.

        Handles the values returned by the
        :py:meth:`~culebra.fitness_function.feature_selection.NumFeats.evaluate`
        method within a
        :py:class:`~culebra.solution.feature_selection.Solution`.
        """

        weights = (-1.0,)
        """Minimize the number of features that a solution has selected."""

        names = ("NF",)
        """Name of the objective."""

        thresholds = [DEFAULT_THRESHOLD]
        """Similarity threshold for fitness comparisons."""

    def evaluate(
        self,
        sol: Solution,
        index: Optional[int] = None,
        representatives: Optional[Sequence[Solution]] = None
    ) -> Tuple[float, ...]:
        """Evaluate a solution.

        :param sol: Solution to be evaluated.
        :type sol: :py:class:`~culebra.solution.feature_selection.Solution`
        :param index: Index where *sol* should be inserted in the
            representatives sequence to form a complete solution for the
            problem. Only used by cooperative problems
        :type index: :py:class:`int`, ignored
        :param representatives: Representative solutions of each species
            being optimized. Only used by cooperative problems
        :type representatives: :py:class:`~collections.abc.Sequence` of
            :py:class:`~culebra.abc.Solution`, ignored
        :return: The fitness of *sol*
        :rtype: :py:class:`tuple` of :py:class:`float`
        """
        # Return the solution's size
        return (sol.num_feats,)


class FeatsProportion(FeatureSelectionFitnessFunction):
    """Dummy single-objective fitness function for testing purposes.

    Return the proportion of selected features chosen by a solution. That is,
    0 if no features are selected or 1 if all the features have been chosen.
    """

    class Fitness(Fitness):
        """Fitness class.

        Handles the values returned by the
        :py:meth:`~culebra.fitness_function.feature_selection.FeatsProportion.evaluate`
        method within a
        :py:class:`~culebra.solution.feature_selection.Solution`.
        """

        weights = (-1.0,)
        """Minimize the proportion of features that a solution has selected."""

        names = ("FP",)
        """Name of the objective."""

        thresholds = [DEFAULT_THRESHOLD]
        """Similarity threshold for fitness comparisons."""

    def evaluate(
        self,
        sol: Solution,
        index: Optional[int] = None,
        representatives: Optional[Sequence[Solution]] = None
    ) -> Tuple[float, ...]:
        """Evaluate a solution.

        :param sol: Solution to be evaluated.
        :type sol: :py:class:`~culebra.solution.feature_selection.Solution`
        :param index: Index where *sol* should be inserted in the
            representatives sequence to form a complete solution for the
            problem. Only used by cooperative problems
        :type index: :py:class:`int`, ignored
        :param representatives: Representative solutions of each species
            being optimized. Only used by cooperative problems
        :type representatives: :py:class:`~collections.abc.Sequence` of
            :py:class:`~culebra.abc.Solution`, ignored
        :return: The fitness of *sol*
        :rtype: :py:class:`tuple` of :py:class:`float`
        """
        # Return the solution's size
        return (float(sol.num_feats)/sol.species.num_feats,)


class KappaIndex(FeatureSelectionFitnessFunction):
    """Single-objective fitness function for classification problems.

    Calculate the Kohen's Kappa index.
    """

    class Fitness(Fitness):
        """Fitness class.

        Handles the values returned by the
        :py:meth:`~culebra.fitness_function.feature_selection.KappaIndex.evaluate`
        method within a
        :py:class:`~culebra.solution.feature_selection.Solution`.
        """

        weights = (1.0,)
        """Maximizes the validation Kappa index."""

        names = ("Kappa",)
        """Name of the objective."""

        thresholds = [DEFAULT_THRESHOLD]
        """Similarity threshold for fitness comparisons."""

    def evaluate(
        self,
        sol: Solution,
        index: Optional[int] = None,
        representatives: Optional[Sequence[Solution]] = None
    ) -> Tuple[float, ...]:
        """Evaluate a solution.

        :param sol: Solution to be evaluated.
        :type sol: :py:class:`~culebra.solution.feature_selection.Solution`
        :param index: Index where *sol* should be inserted in the
            representatives sequence to form a complete solution for the
            problem. Only used by cooperative problems
        :type index: :py:class:`int`, ignored
        :param representatives: Representative solutions of each species
            being optimized. Only used by cooperative problems
        :type representatives: :py:class:`~collections.abc.Sequence` of
            :py:class:`~culebra.abc.Solution`, ignored
        :return: The fitness of *sol*
        :rtype: :py:class:`tuple` of :py:class:`float`
        """
        kappa = 0

        if sol.features.size > 0:
            # Get the training and test data
            training_data, test_data = self._final_training_test_data()

            # Train and get the outputs for the validation data
            outputs_pred = self.classifier.fit(
                training_data.inputs[:, sol.features],
                training_data.outputs
            ).predict(test_data.inputs[:, sol.features])
            kappa = cohen_kappa_score(test_data.outputs, outputs_pred)

        return (kappa,)


class Accuracy(FeatureSelectionFitnessFunction):
    """Single-objective fitness function for classification problems.

    Calculate the accuracy.
    """

    class Fitness(Fitness):
        """Fitness class.

        Handles the values returned by the
        :py:meth:`~culebra.fitness_function.feature_selection.Accuracy.evaluate`
        method within a
        :py:class:`~culebra.solution.feature_selection.Solution`.
        """

        weights = (1.0,)
        """Maximizes the validation accuracy."""

        names = ("Accuracy",)
        """Name of the objective."""

        thresholds = [DEFAULT_THRESHOLD]
        """Similarity threshold for fitness comparisons."""

    def evaluate(
        self,
        sol: Solution,
        index: Optional[int] = None,
        representatives: Optional[Sequence[Solution]] = None
    ) -> Tuple[float, ...]:
        """Evaluate a solution.

        :param sol: Solution to be evaluated.
        :type sol: :py:class:`~culebra.solution.feature_selection.Solution`
        :param index: Index where *sol* should be inserted in the
            representatives sequence to form a complete solution for the
            problem. Only used by cooperative problems
        :type index: :py:class:`int`, ignored
        :param representatives: Representative solutions of each species
            being optimized. Only used by cooperative problems
        :type representatives: :py:class:`~collections.abc.Sequence` of
            :py:class:`~culebra.abc.Solution`, ignored
        :return: The fitness of *sol*
        :rtype: :py:class:`tuple` of :py:class:`float`
        """
        accuracy = 0

        if sol.features.size > 0:
            # Get the training and test data
            training_data, test_data = self._final_training_test_data()

            # Train and get the outputs for the validation data
            outputs_pred = self.classifier.fit(
                training_data.inputs[:, sol.features],
                training_data.outputs
            ).predict(test_data.inputs[:, sol.features])
            accuracy = accuracy_score(test_data.outputs, outputs_pred)

        return (accuracy,)


class KappaNumFeats(KappaIndex, NumFeats):
    """Bi-objective fitness class for feature selection.

    Maximizes the Kohen's Kappa index and minimizes the number of features
    that a solution has selected.
    """

    class Fitness(Fitness):
        """Fitness class.

        Handles the values returned by the
        :py:meth:`~culebra.fitness_function.feature_selection.KappaNumFeats.evaluate`
        method within a
        :py:class:`~culebra.solution.feature_selection.Solution`.
        """

        weights = KappaIndex.Fitness.weights + NumFeats.Fitness.weights
        """Maximizes the Kohen's Kappa index and minimizes the number of
        features that a solution has selected.
        """

        names = KappaIndex.Fitness.names + NumFeats.Fitness.names
        """Name of the objectives."""

        thresholds = (
            KappaIndex.Fitness.thresholds + NumFeats.Fitness.thresholds
        )
        """Similarity thresholds for fitness comparisons."""

    def evaluate(
        self,
        sol: Solution,
        index: Optional[int] = None,
        representatives: Optional[Sequence[Solution]] = None
    ) -> Tuple[float, ...]:
        """Evaluate a solution.

        :param sol: Solution to be evaluated.
        :type sol: :py:class:`~culebra.solution.feature_selection.Solution`
        :param index: Index where *sol* should be inserted in the
            representatives sequence to form a complete solution for the
            problem. Only used by cooperative problems
        :type index: :py:class:`int`, ignored
        :param representatives: Representative solutions of each species
            being optimized. Only used by cooperative problems
        :type representatives: :py:class:`~collections.abc.Sequence` of
            :py:class:`~culebra.abc.Solution`, ignored
        :return: The fitness of *sol*
        :rtype: :py:class:`tuple` of :py:class:`float`
        """
        return KappaIndex.evaluate(self, sol) + NumFeats.evaluate(self, sol)


class KappaFeatsProp(KappaIndex, FeatsProportion):
    """Bi-objective fitness class for feature selection.

    Maximizes the Kohen's Kappa index and minimizes the proportion of features
    that a solution has selected.
    """

    class Fitness(Fitness):
        """Fitness class.

        Handles the values returned by the
        :py:meth:`~culebra.fitness_function.feature_selection.KappaFeatsProp.evaluate`
        method within a
        :py:class:`~culebra.solution.feature_selection.Solution`.
        """

        weights = KappaIndex.Fitness.weights + FeatsProportion.Fitness.weights
        """Maximizes the Kohen's Kappa index and minimizes the propotion of
        features that a solution has selected.
        """

        names = KappaIndex.Fitness.names + FeatsProportion.Fitness.names
        """Name of the objectives."""

        thresholds = (
            KappaIndex.Fitness.thresholds + FeatsProportion.Fitness.thresholds
        )
        """Similarity thresholds for fitness comparisons."""

    def evaluate(
        self,
        sol: Solution,
        index: Optional[int] = None,
        representatives: Optional[Sequence[Solution]] = None
    ) -> Tuple[float, ...]:
        """Evaluate a solution.

        :param sol: Solution to be evaluated.
        :type sol: :py:class:`~culebra.solution.feature_selection.Solution`
        :param index: Index where *sol* should be inserted in the
            representatives sequence to form a complete solution for the
            problem. Only used by cooperative problems
        :type index: :py:class:`int`, ignored
        :param representatives: Representative solutions of each species
            being optimized. Only used by cooperative problems
        :type representatives: :py:class:`~collections.abc.Sequence` of
            :py:class:`~culebra.abc.Solution`, ignored
        :return: The fitness of *sol*
        :rtype: :py:class:`tuple` of :py:class:`float`
        """
        return (
            KappaIndex.evaluate(self, sol) +
            FeatsProportion.evaluate(self, sol)
        )


class AccuracyNumFeats(Accuracy, NumFeats):
    """Bi-objective fitness class for feature selection.

    Maximizes the accuracy and minimizes the number of features that a
    solution has selected.
    """

    class Fitness(Fitness):
        """Fitness class.

        Handles the values returned by the
        :py:meth:`~culebra.fitness_function.feature_selection.AccuracyNumFeats.evaluate`
        method within a
        :py:class:`~culebra.solution.feature_selection.Solution`.
        """

        weights = Accuracy.Fitness.weights + NumFeats.Fitness.weights
        """Maximizes the accuracy and minimizes the number of features that a
        solution has selected.
        """

        names = Accuracy.Fitness.names + NumFeats.Fitness.names
        """Name of the objectives."""

        thresholds = (
            Accuracy.Fitness.thresholds + NumFeats.Fitness.thresholds
        )
        """Similarity thresholds for fitness comparisons."""

    def evaluate(
        self,
        sol: Solution,
        index: Optional[int] = None,
        representatives: Optional[Sequence[Solution]] = None
    ) -> Tuple[float, ...]:
        """Evaluate a solution.

        :param sol: Solution to be evaluated.
        :type sol: :py:class:`~culebra.solution.feature_selection.Solution`
        :param index: Index where *sol* should be inserted in the
            representatives sequence to form a complete solution for the
            problem. Only used by cooperative problems
        :type index: :py:class:`int`, ignored
        :param representatives: Representative solutions of each species
            being optimized. Only used by cooperative problems
        :type representatives: :py:class:`~collections.abc.Sequence` of
            :py:class:`~culebra.abc.Solution`, ignored
        :return: The fitness of *sol*
        :rtype: :py:class:`tuple` of :py:class:`float`
        """
        return Accuracy.evaluate(self, sol) + NumFeats.evaluate(self, sol)


class AccuracyFeatsProp(Accuracy, FeatsProportion):
    """Bi-objective fitness class for feature selection.

    Maximizes the accuracy and minimizes the proportion of features that a
    solution has selected.
    """

    class Fitness(Fitness):
        """Fitness class.

        Handles the values returned by the
        :py:meth:`~culebra.fitness_function.feature_selection.AccuracyFeatsProp.evaluate`
        method within a
        :py:class:`~culebra.solution.feature_selection.Solution`.
        """

        weights = Accuracy.Fitness.weights + FeatsProportion.Fitness.weights
        """Maximizes the accuracy and minimizes the proportion of features that
        a solution has selected.
        """

        names = Accuracy.Fitness.names + FeatsProportion.Fitness.names
        """Name of the objectives."""

        thresholds = (
            Accuracy.Fitness.thresholds + FeatsProportion.Fitness.thresholds
        )
        """Similarity thresholds for fitness comparisons."""

    def evaluate(
        self,
        sol: Solution,
        index: Optional[int] = None,
        representatives: Optional[Sequence[Solution]] = None
    ) -> Tuple[float, ...]:
        """Evaluate a solution.

        :param sol: Solution to be evaluated.
        :type sol: :py:class:`~culebra.solution.feature_selection.Solution`
        :param index: Index where *sol* should be inserted in the
            representatives sequence to form a complete solution for the
            problem. Only used by cooperative problems
        :type index: :py:class:`int`, ignored
        :param representatives: Representative solutions of each species
            being optimized. Only used by cooperative problems
        :type representatives: :py:class:`~collections.abc.Sequence` of
            :py:class:`~culebra.abc.Solution`, ignored
        :return: The fitness of *sol*
        :rtype: :py:class:`tuple` of :py:class:`float`
        """
        return (
            Accuracy.evaluate(self, sol) +
            FeatsProportion.evaluate(self, sol)
        )


# Exported symbols for this module
__all__ = [
    'NumFeats',
    'FeatsProportion',
    'KappaIndex',
    'Accuracy',
    'KappaNumFeats',
    'KappaFeatsProp',
    'AccuracyNumFeats',
    'AccuracyFeatsProp',
]
