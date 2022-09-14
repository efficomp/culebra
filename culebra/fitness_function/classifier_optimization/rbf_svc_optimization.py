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

"""Optimize the hyperparameters of a SVC with an RBF kernel function."""

from __future__ import annotations
from typing import Optional, Tuple
from collections.abc import Sequence
from sklearn.base import ClassifierMixin
from sklearn.svm import SVC
from sklearn.metrics import cohen_kappa_score
from culebra.base import (
    Dataset,
    Fitness,
    FitnessFunction,
    check_instance
)
from culebra.genotype.classifier_optimization import Individual


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2021, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.1.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


DEFAULT_THRESHOLD = 0.01
"""Default similarity threshold for fitnesses."""


class RBFSVCFitnessFunction(FitnessFunction):
    """Abstract base class fitness function for RBF SVC optimization."""

    def __init__(
        self,
        training_data: Dataset,
        test_data: Optional[Dataset] = None,
        test_prop: Optional[float] = None,
        classifier: ClassifierMixin = SVC(kernel='rbf')
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
        :param test_prop: A real value in (0, 1) or :py:data:`None`. Defaults
            to :py:data:`None`
        :type test_prop: :py:class:`float`, optional
        :param classifier: The classifier
        :type classifier: :py:class:`~sklearn.svm.SVC` with
            RBF kernels, optional
        """
        # Init the superclass
        super().__init__(training_data, test_data, test_prop, classifier)

    @property
    def classifier(self) -> SVC:
        """Get and set the classifier applied within this wrapper.

        :getter: Return the classifier
        :setter: Set a new classifier
        :type: Any subclass of :py:class:`~sklearn.svm.SVC` with
            RBF kernels
        :raises TypeError: If set to a value which is not a
            :py:class:`~sklearn.svm.SVC`
        :raises ValueError: If the classifier has not RBF kernels
        """
        return self._classifier

    @classifier.setter
    def classifier(self, value: SVC) -> None:
        """Set a classifier.

        :param value: The classifier
        :type value: Any subclass of :py:class:`~sklearn.svm.SVC` with
            RBF kernels
        :raises TypeError: If *value* is not a :py:class:`~sklearn.svm.SVC`
        :raises ValueError: If the classifier has not RBF kernels
        """
        self._classifier = check_instance(
            value, "classifier", cls=SVC
        )

        if self._classifier.kernel != 'rbf':
            raise ValueError(
                f"The classifier has not RBF kernels: {value}"
            )


class C(RBFSVCFitnessFunction):
    """Minimization of the C hyperparameter of RBF SVCs."""

    class Fitness(Fitness):
        """Fitness class.

        Handles the values returned by the
        :py:meth:`~fitness_function.classifier_optimization.C.evaluate`
        method within an
        :py:class:`~genotype.classifier_optimization.Individual`.
        """

        weights = (-1.0,)
        """Minimize C."""

        names = ("C",)
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
        :type ind: :py:class:`~genotype.classifier_optimization.Individual`
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
        return (ind.values.C,)


class KappaIndex(RBFSVCFitnessFunction):
    """Maximize the Kohen's Kappa index."""

    class Fitness(Fitness):
        """Fitness class.

        Handles the values returned by the
        :py:meth:`~fitness_function.classifier_optimization.C.evaluate`
        method within an
        :py:class:`~genotype.classifier_optimization.Individual`.
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
        :type ind: :py:class:`~genotype.classifier_optimization.Individual`
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
        # Set the classifier hyperparameters
        hyperparams = ind.values
        self.classifier.C = hyperparams.C
        self.classifier.gamma = hyperparams.gamma

        # Get the training and test data
        training_data, test_data = self._final_training_test_data()

        # Train and get the outputs for the validation data
        outputs_pred = self.classifier.fit(
            training_data.inputs, training_data.outputs
        ).predict(test_data.inputs)
        kappa = cohen_kappa_score(test_data.outputs, outputs_pred)

        return (kappa,)


class KappaC(KappaIndex, C):
    """Bi-objective fitness class for feature selection.

    Maximizes the Kohen's Kappa index and minimizes the C regularization
    htyperparameter.
    """

    class Fitness(Fitness):
        """Fitness class.

        Handles the values returned by the
        :py:meth:`~fitness_function.classifier_optimization.C.evaluate`
        method within an
        :py:class:`~genotype.classifier_optimization.Individual`.
        """

        weights = KappaIndex.Fitness.weights + C.Fitness.weights
        """Maximizes the Kohen's Kappa index and minimizes the C regularization
        hyperparameter.
        """

        names = KappaIndex.Fitness.names + C.Fitness.names
        """Name of the objectives."""

        thresholds = (
            KappaIndex.Fitness.thresholds + C.Fitness.thresholds
        )
        """Similarity threshold for fitness comparisons."""

    def evaluate(
        self,
        ind: Individual,
        index: Optional[int] = None,
        representatives: Optional[Sequence[Individual]] = None
    ) -> Tuple[float, ...]:
        """Evaluate an individual.

        :param ind: Individual to be evaluated.
        :type ind: :py:class:`~genotype.classifier_optimization.Individual`
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
        return KappaIndex.evaluate(self, ind) + C.evaluate(self, ind)


# Exported symbols for this module
__all__ = ['RBFSVCFitnessFunction', 'C', 'KappaIndex', 'KappaC']
