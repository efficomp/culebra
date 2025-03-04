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

"""Abstract base fitness functions.

This sub-module provides several abstract classes that help defining other
fitness functions. The following classes are provided:

  * :py:class:`~culebra.fitness_function.abc.DatasetFitnessFunction`: Allows
    the definition of dataset-related fitness functions.

  * :py:class:`~culebra.fitness_function.abc.ClassificationFitnessFunction`:
    Lets defining classification-related fitness functions

  * :py:class:`~culebra.fitness_function.abc.FeatureSelectionFitnessFunction`:
    Designed to support feature selection problems

  * :py:class:`~culebra.fitness_function.abc.RBFSVCFitnessFunction`:
    Is centered on the hyperparameters optimization of SVM-based classifiers
    with RBF kernels.

  * :py:class:`~culebra.fitness_function.abc.CooperativeFSFitnessFunction`:
    Abstract base class for all the fitness functions of cooperative FS
    problems.

  * :py:class:`~culebra.fitness_function.abc.CooperativeRBFSVCFSFitnessFunction`:
    Abstract base class for all the fitness functions of cooperative FS
    problems using RBF SVM classifiers.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Tuple, Optional
from copy import deepcopy

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.svm import SVC

from culebra.abc import FitnessFunction, Solution
from culebra.checker import check_float, check_instance
from culebra.fitness_function import DEFAULT_CLASSIFIER
from culebra.solution.feature_selection import Species as FSSpecies
from culebra.tools import Dataset


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2025, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.4.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


class DatasetFitnessFunction(FitnessFunction):
    """Abstract base fitness function for dataset evaluation problems."""

    def __init__(
        self,
        training_data: Dataset,
        test_data: Optional[Dataset] = None,
        test_prop: Optional[float] = None,
    ) -> None:
        """Construct a fitness function.

        If *test_data* are provided, the whole *training_data* are used to
        train. Otherwise, if *test_prop* is provided, *training_data* are
        split (stratified) into training and test data. Finally, if both
        *test_data* and *test_prop* are omitted, *training_data* are also used
        to test.

        :param training_data: The training dataset
        :type training_data: :py:class:`~culebra.tools.Dataset`
        :param test_data: The test dataset, defaults to :py:data:`None`
        :type test_data: :py:class:`~culebra.tools.Dataset`, optional
        :param test_prop: A real value in [0, 1] or :py:data:`None`. Defaults
            to :py:data:`None`
        :type test_prop: :py:class:`float`, optional
        """
        # Init the superclass
        super().__init__()

        # Set the attributes to default values
        self.training_data = training_data
        self.test_data = test_data
        self.test_prop = test_prop

    @property
    def training_data(self) -> Dataset:
        """Get and set the training dataset.

        :getter: Return the training dataset
        :setter: Set a new training dataset
        :type: :py:class:`~culebra.tools.Dataset`
        :raises TypeError: If set to an invalid dataset
        """
        return self._training_data

    @training_data.setter
    def training_data(self, value: Dataset) -> None:
        """Set a new training dataset.

        :param value: A new training dataset
        :type value: :py:class:`~culebra.tools.Dataset`
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
        :type: :py:class:`~culebra.tools.Dataset`
        :raises TypeError: If set to an invalid dataset
        """
        return self._test_data

    @test_data.setter
    def test_data(self, value: Dataset | None) -> None:
        """Set a new test dataset.

        :param value: A new test dataset
        :type value: :py:class:`~culebra.tools.Dataset` or :py:data:`None`
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

    def _final_training_test_data(self) -> Tuple[Dataset, Dataset]:
        """Get the final training and test data.

        If *test_data* is not :py:data:`None`, the whole *training_data* are
        used to train. Otherwise, if *test_prop* is not :py:data:`None`,
        *training_data* are split (stratified) into training and test data.
        Finally, if both *test_data* and *test_prop* are :py:data:`None`,
        *training_data* are also used to test.

        :return: The final training and test datasets
        :rtype: :py:class:`tuple` of :py:class:`~culebra.tools.Dataset`
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

    def __copy__(self) -> DatasetFitnessFunction:
        """Shallow copy the fitness function."""
        cls = self.__class__
        result = cls(self.training_data)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo: dict) -> DatasetFitnessFunction:
        """Deepcopy the fitness function.

        :param memo: Fitness function attributes
        :type memo: :py:class:`dict`
        :return: A deep copy of the fitness function
        :rtype:
            :py:class:`~culebra.feature_selection.abc.DatasetFitnessFunction`
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


class ClassificationFitnessFunction(DatasetFitnessFunction):
    """Abstract base fitness function for classification problems."""

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
        :type training_data: :py:class:`~culebra.tools.Dataset`
        :param test_data: The test dataset, defaults to :py:data:`None`
        :type test_data: :py:class:`~culebra.tools.Dataset`, optional
        :param test_prop: A real value in [0, 1] or :py:data:`None`. Defaults
            to :py:data:`None`
        :type test_prop: :py:class:`float`, optional
        :param classifier: The classifier, defaults to
            :py:attr:`~culebra.fitness_function.DEFAULT_CLASSIFIER`
        :type classifier: :py:class:`~sklearn.base.ClassifierMixin`,
            optional
        """
        # Init the superclass
        super().__init__(training_data, test_data, test_prop)
        self.classifier = classifier

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


class FeatureSelectionFitnessFunction(ClassificationFitnessFunction):
    """Abstract base fitness function for feature selection problems."""

    @property
    def num_nodes(self) -> int:
        """Return the problem graph's number of nodes for ACO-based trainers.

        :return: The problem graph's number of nodes
        :rtype: :py:class:`int`
        """
        return self.training_data.num_feats

    def heuristic(self, species: FSSpecies) -> Sequence[np.ndarray, ...]:
        """Get the heuristic matrices for ACO-based trainers.

        :param species: Species constraining the problem solutions
        :type species: :py:class:`~culebra.solution.feature_selection.Species`
        :raises TypeError: If *species* is not an instance of
            :py:class:`~culebra.solution.feature_selection.Species`

        :return: A tuple with only one heuristic matrix. Arcs between
            selectable features have a heuristic value of 1, while arcs
            involving any non-selectable feature or arcs from a feature to
            itself have a heuristic value of 0.
        :rtype: :py:class:`~collections.abc.Sequence` of
            :py:class:`~numpy.ndarray`
        """
        check_instance(species, "species", cls=FSSpecies)

        num_feats = species.num_feats

        # All the features should be considered
        heuristic = np.ones((num_feats, num_feats))

        # Ignore features with an index lower than min_feat
        min_feat = species.min_feat
        if min_feat > 0:
            for feat in range(num_feats):
                for ignored in range(min_feat):
                    heuristic[feat][ignored] = 0
                    heuristic[ignored][feat] = 0

        # Ignore features with an index greater than max_feat
        max_feat = species.max_feat
        if max_feat < num_feats - 1:
            for feat in range(num_feats):
                for ignored in range(max_feat + 1, num_feats):
                    heuristic[feat][ignored] = 0
                    heuristic[ignored][feat] = 0

        # The distance from a feature to itself is also ignored
        for index in range(min_feat, max_feat+1):
            heuristic[index][index] = 0

        return (heuristic, )


class RBFSVCFitnessFunction(ClassificationFitnessFunction):
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
        :type training_data: :py:class:`~culebra.tools.Dataset`
        :param test_data: The test dataset, defaults to :py:data:`None`
        :type test_data: :py:class:`~culebra.tools.Dataset`, optional
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
        """Get and set the classifier applied within this fitness function.

        :getter: Return the classifier
        :setter: Set a new classifier
        :type: Any subclass of :py:class:`~sklearn.svm.SVC` with RBF kernels
        :raises TypeError: If set to a value which is not an
            :py:class:`~sklearn.svm.SVC`
        :raises ValueError: If the classifier has not RBF kernels
        """
        return self._classifier

    @classifier.setter
    def classifier(self, value: SVC) -> None:
        """Set a classifier.

        :param value: The classifier
        :type value: Any subclass of :py:class:`~sklearn.svm.SVC` with RBF
            kernels
        :raises TypeError: If *value* is not an :py:class:`~sklearn.svm.SVC`
        :raises ValueError: If the classifier has not RBF kernels
        """
        self._classifier = check_instance(
            value, "classifier", cls=SVC
        )

        if self._classifier.kernel != 'rbf':
            raise ValueError(
                f"The classifier has not RBF kernels: {value}"
            )


class CooperativeFSFitnessFunction(FitnessFunction):
    """Abstract base class fitness function for cooperative FS problems."""

    def construct_solutions(
        self,
        sol: Solution,
        index: Optional[int] = None,
        representatives: Optional[Sequence[Solution]] = None
    ) -> Tuple[Solution, ...]:
        """Assemble the solution and representatives.

           This fitness function assumes that:

             * *representatives[0]*: Codes the SVC hyperparameters
               (C and gamma). Thus, it is an instance of
               :py:class:`culebra.solution.parameter_optimization.Solution`
             * *representatives[1:]*: The remaining solutions code the
               features selected, each solution a different range of
               features. All of them are instances of
               :py:class:`culebra.solution.feature_selection.Solution`

        :param sol: Solution to be evaluated.
        :type sol: :py:class:`~culebra.abc.Solution`
        :param index: Index where *sol* should be inserted in the
            representatives sequence to form a complete solution for the
            problem
        :type index: :py:class:`int`
        :param representatives: Representative solutions of each species
            being optimized
        :type representatives: :py:class:`~collections.abc.Sequence` of
            :py:class:`~culebra.abc.Solution`, ignored
        :return: The solutions to the different problems solved cooperatively
        :rtype: :py:class:`tuple` of py:class:`culebra.abc.Solution`
        """
        # Number of representatives
        num_representatives = len(representatives)

        # Hyperparameters solution
        sol_hyperparams = sol if index == 0 else representatives[0]

        # Prototype solution for the final solution containing all the
        # features
        prototype_sol_features = sol if index == 1 else representatives[1]

        # All the features
        all_the_features = []

        # Number of features
        number_features = prototype_sol_features.species.num_feats

        # Update the features and feature min and max indices
        for repr_index in range(1, num_representatives):
            # Choose thge correct solution
            the_sol = (
                sol if repr_index == index else representatives[repr_index]
            )
            # Get the features
            all_the_features += list(the_sol.features)

        # Features solution class
        sol_features_cls = prototype_sol_features.__class__

        # Features solution species class
        sol_features_species_cls = prototype_sol_features.species.__class__

        # Features solution species
        sol_features_species = sol_features_species_cls(number_features)

        # Features solution
        sol_features = sol_features_cls(
            species=sol_features_species,
            fitness_cls=self.Fitness,
            features=all_the_features
        )

        return (sol_features, sol_hyperparams)


class CooperativeRBFSVCFSFitnessFunction(
    CooperativeFSFitnessFunction,
    RBFSVCFitnessFunction
):
    """Abstract base class for cooperative FS problems using an RBF SVC classifier."""


# Exported symbols for this module
__all__ = [
    'DatasetFitnessFunction',
    'ClassificationFitnessFunction',
    'FeatureSelectionFitnessFunction',
    'RBFSVCFitnessFunction',
    'CooperativeFSFitnessFunction',
    'CooperativeRBFSVCFSFitnessFunction'
]
