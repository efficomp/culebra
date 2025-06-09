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

  * :py:class:`~culebra.fitness_function.abc.DatasetScorer`: Allows
    the definition of dataset-related fitness functions.

  * :py:class:`~culebra.fitness_function.abc.ClassificationScorer`:
    Lets defining dataset classification-related fitness functions

  * :py:class:`~culebra.fitness_function.abc.ClassificationFSScorer`:
    Designed to support feature selection on classification problems

  * :py:class:`~culebra.fitness_function.abc.RBFSVCScorer`:
    Is centered on the hyperparameters optimization of SVM-based classifiers
    with RBF kernels.

  * :py:class:`~culebra.fitness_function.abc.CooperativeFSScorer`:
    Abstract base class for all the fitness functions of cooperative FS
    problems.

  * :py:class:`~culebra.fitness_function.abc.CooperativeRBFSVCFSScorer`:
    Abstract base class for all the fitness functions of cooperative FS
    problems using RBF SVM classifiers.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import Tuple, Optional, Any
from copy import deepcopy

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer

from culebra.abc import FitnessFunction, Solution
from culebra.checker import check_int, check_float, check_instance
from culebra.fitness_function import DEFAULT_CLASSIFIER, DEFAULT_CV_FOLDS
from culebra.solution.feature_selection import Species as FSSpecies
from culebra.tools import Dataset


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2025, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.4.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


class DatasetScorer(FitnessFunction):
    """Abstract base fitness function for dataset evaluation problems."""

    def __init__(
        self,
        training_data: Dataset,
        test_data: Optional[Dataset] = None,
        test_prop: Optional[float] = None,
        cv_folds: Optional[int] = None
    ) -> None:
        """Construct a fitness function.

        If *test_data* are provided, the whole *training_data* are used to
        train. Otherwise, if *test_prop* is provided, *training_data* are
        split (stratified) into training and test data each time
        :py:meth:`~culebra.FitnessFunction.DatasetScorer.evaluate` is
        called and a Monte Carlo cross validation is applied. Finally, if both
        *test_data* and *test_prop* are omitted, a *k*-fold cross-validation
        is applied.

        :param training_data: The training dataset
        :type training_data: :py:class:`~culebra.tools.Dataset`
        :param test_data: The test dataset, defaults to :py:data:`None`
        :type test_data: :py:class:`~culebra.tools.Dataset`, optional
        :param test_prop: A real value in (0, 1) or :py:data:`None`. Defaults
            to :py:data:`None`
        :type test_prop: :py:class:`float`, optional
        :param cv_folds: The number of folds for *k*-fold cross-validation.
            If omitted, :py:attr:`~culebra.fitness_function.DEFAULT_CV_FOLDS`
            is used. Defaults to :py:data:`None`
        :type cv_folds: :py:class:`int`, optional
        """
        # Init the superclass
        FitnessFunction.__init__(self)

        # Set the attributes to default values
        self.training_data = training_data
        self.test_data = test_data
        self.test_prop = test_prop
        self.cv_folds = cv_folds

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
            (0, 1) or :py:data:`None` is expected
        :type: :py:class:`float`
        :raises TypeError: If set to a value which is not a real number
        :raises ValueError: If set to a value which is not in (0, 1)
        """
        return self._test_prop

    @test_prop.setter
    def test_prop(self, value: float | None) -> None:
        """Set a value for the proportion of data used to test.

        :param value: A real value in (0, 1) or :py:data:`None`
        :type value: :py:class:`float` or :py:data:`None`
        :raises TypeError: If *value* is not a real number
        :raises ValueError: If *value* is not in (0, 1)
        """
        self._test_prop = None if value is None else check_float(
            value, "test proportion", gt=0, lt=1
        )

    @property
    def cv_folds(self) -> int | None:
        """Get and set the number of cross-validation folds.

        :getter: Return the number of cross-validation folds
        :setter: Set a new value for the number of cross-validation folds. A
            positive integer value or :py:data:`None` is expected
        :type: :py:class:`int`
        :raises TypeError: If set to a value which is not an integer value
        :raises ValueError: If set to a value which is not positive
        """
        return (
            self._cv_folds if self._cv_folds is not None else DEFAULT_CV_FOLDS
        )

    @cv_folds.setter
    def cv_folds(self, value: int | None) -> None:
        """Set a value for the number of cross-validation folds.

        :param value: A positive integer value in (0, 1) or :py:data:`None`
        :type value: :py:class:`int` or :py:data:`None`
        :raises TypeError: If *value* is not an integer value
        :raises ValueError: If *value* is not positive
        """
        self._cv_folds = None if value is None else check_int(
            value, "number of cross-validation folds", gt=0
        )

    @property
    def is_noisy(self) -> int:
        """Return :py:attr:`True` if the fitness function is noisy."""
        if self.test_data is None and self.test_prop is not None:
            return True

        return False

    @property
    @abstractmethod
    def _worst_score(self) -> float:
        """Worst achievable score.

        This property must be overridden by subclasses to return a correct
        value.

        :raises NotImplementedError: if has not been overridden
        :type: :py:class:`float`
        """
        raise NotImplementedError(
            "The path property has not been implemented in the "
            f"{self.__class__.__name__} class"
        )

    @staticmethod
    @abstractmethod
    def _score(
        outputs: Sequence[float],
        outputs_pred: Sequence[float],
        **kwargs: Any
    ) -> float:
        """Score function to be used in the evaluation.

        This method must be overridden by subclasses to return a correct
        value.
        """
        raise NotImplementedError(
            "The scorer method has not been implemented")

    def _final_training_test_data(
        self,
        sol: Solution
    ) -> Tuple[Dataset, Dataset]:
        """Get the final training and test data.

        :param sol: Solution to be evaluated. It may influence the final
            datasets
        :type sol: :py:class:`~culebra.abc.Solution`

        :return: The final training and test datasets
        :rtype: :py:class:`tuple` of :py:class:`~culebra.tools.Dataset`
        """
        return self.training_data, self.test_data

    @abstractmethod
    def _evaluate_train_test(
        self,
        training_data: Dataset,
        test_data: Dataset
    ) -> Tuple[float, ...]:
        """Evaluate a solution.

        This method must be overridden by subclasses to return a correct
        value.

        :param training_data: The training dataset
        :type training_data: :py:class:`~culebra.tools.Dataset`
        :param test_data: The test dataset
        :type test_data: :py:class:`~culebra.tools.Dataset`
        :raises NotImplementedError: if has not been overridden
        :return: The fitness values for *sol*
        :rtype: :py:class:`tuple` of :py:class:`float`
        """
        raise NotImplementedError(
            "The _evaluate_train_test method has not been implemented in the "
            f"{self.__class__.__name__} class")

    @abstractmethod
    def _evaluate_mccv(
        self,
        training_data: Dataset
    ) -> Tuple[float, ...]:
        """Evaluate a solution.

        The *training_data* are split (stratified) into training and test data
        according to
        :py:attr:`~culebra.fitness_function.abc.DatasetScorer.test_prop`
        each time the solution is evalueted and a Monte Carlo cross-validation
        is applied.

        This method must be overridden by subclasses to return a correct
        value.

        :param training_data: The training dataset
        :type training_data: :py:class:`~culebra.tools.Dataset`
        :return: The fitness values for *sol*
        :rtype: :py:class:`tuple` of :py:class:`float`
        """
        raise NotImplementedError(
            "The _evaluate_mccv method has not been implemented in the "
            f"{self.__class__.__name__} class")

    @abstractmethod
    def _evaluate_kfcv(
        self,
        training_data: Dataset,
    ) -> Tuple[float, ...]:
        """Evaluate a solution.

        A *k*-fold cross-validation is applied using the *training_data* with
        :py:attr:`~culebra.fitness_function.abc.DatasetScorer.cv_folds`
        folds.

        This method must be overridden by subclasses to return a correct
        value.

        :param training_data: The training dataset
        :type training_data: :py:class:`~culebra.tools.Dataset`
        :raises NotImplementedError: if has not been overridden
        :return: The fitness values for *sol*
        :rtype: :py:class:`tuple` of :py:class:`float`
        """
        raise NotImplementedError(
            "The _evaluate_kfcv method has not been implemented in the "
            f"{self.__class__.__name__} class")

    def evaluate(
        self,
        sol: Solution,
        index: Optional[int] = None,
        representatives: Optional[Sequence[Solution]] = None
    ) -> Tuple[float, ...]:
        """Evaluate a solution.

        :param sol: Solution to be evaluated.
        :type sol: :py:class:`~culebra.abc.Solution`
        :param index: Index where *sol* should be inserted in the
            representatives sequence to form a complete solution for the
            problem. Only used by cooperative problems
        :type index: :py:class:`int`, optional
        :param representatives: Representative solutions of each species
            being optimized. Only used by cooperative problems
        :type representatives: A :py:class:`~collections.abc.Sequence`
            containing instances of :py:class:`~culebra.abc.Solution`,
            optional
        :return: The fitness values for *sol*
        :rtype: :py:class:`tuple` of :py:class:`float`
        """
        # Obtain the final training and test data
        training_data, test_data = self._final_training_test_data(sol)

        # If test data are available, train with the whole training data
        # and test with the test data
        if test_data is not None:
            return self._evaluate_train_test(training_data, test_data)
        # If not, if a test porportion is given, apply Monte Carlo
        # cross-validation
        elif self.test_prop is not None:
            return self._evaluate_mccv(training_data)
        # Else, apply k-fold cross-validation
        else:
            return self._evaluate_kfcv(training_data)

    def __copy__(self) -> DatasetScorer:
        """Shallow copy the fitness function."""
        cls = self.__class__
        result = cls(self.training_data)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo: dict) -> DatasetScorer:
        """Deepcopy the fitness function.

        :param memo: Fitness function attributes
        :type memo: :py:class:`dict`
        :return: A deep copy of the fitness function
        :rtype:
            :py:class:`~culebra.feature_selection.abc.DatasetScorer`
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


class ClassificationScorer(DatasetScorer):
    """Abstract base fitness function for dataset classification problems."""

    def __init__(
        self,
        training_data: Dataset,
        test_data: Optional[Dataset] = None,
        test_prop: Optional[float] = None,
        cv_folds: Optional[int] = None,
        classifier: Optional[ClassifierMixin] = None
    ) -> None:
        """Construct a fitness function.

        If *test_data* are provided, the whole *training_data* are used to
        train. Otherwise, if *test_prop* is provided, *training_data* are
        split (stratified) into training and test data each time
        :py:meth:`~culebra.FitnessFunction.ClassificationScorer.evaluate`
        is called and a Monte Carlo cross validation is applied. Finally, if
        both *test_data* and *test_prop* are omitted, a *k*-fold
        cross-validation is applied.

        :param training_data: The training dataset
        :type training_data: :py:class:`~culebra.tools.Dataset`
        :param test_data: The test dataset, defaults to :py:data:`None`
        :type test_data: :py:class:`~culebra.tools.Dataset`, optional
        :param test_prop: A real value in (0, 1) or :py:data:`None`. Defaults
            to :py:data:`None`
        :type test_prop: :py:class:`float`, optional
        :param cv_folds: The number of folds for *k*-fold cross-validation.
            If omitted, :py:attr:`~culebra.fitness_function.DEFAULT_CV_FOLDS`
            is used. Defaults to :py:data:`None`
        :type cv_folds: :py:class:`int`, optional
        :param classifier: The classifier. If set to :py:data:`None`,
            :py:attr:`~culebra.fitness_function.DEFAULT_CLASSIFIER`
            will be used. Defaults to :py:data:`None`
        :type classifier: :py:class:`~sklearn.base.ClassifierMixin`,
            optional
        """
        # Init the superclass
        DatasetScorer.__init__(
            self, training_data, test_data, test_prop, cv_folds
        )
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
    def classifier(self, value: ClassifierMixin | None) -> None:
        """Set a classifier.

        :param value: The classifier. If set to :py:data:`None`,
            :py:attr:`~culebra.fitness_function.DEFAULT_CLASSIFIER`
            is chosen
        :type value: Any subclass of :py:class:`~sklearn.base.ClassifierMixin`
        :raises TypeError: If *value* is not a classifier
        """
        self._classifier = (
            DEFAULT_CLASSIFIER() if value is None else check_instance(
                value, "classifier", cls=ClassifierMixin
            )
        )

    def _evaluate_train_test(
        self,
        training_data: Dataset,
        test_data: Dataset
    ) -> Tuple[float, ...]:
        """Evaluate a solution.

        This method must be overridden by subclasses to return a correct
        value.

        :param training_data: The training dataset
        :type training_data: :py:class:`~culebra.tools.Dataset`
        :param test_data: The test dataset
        :type test_data: :py:class:`~culebra.tools.Dataset`
        :return: The fitness values for *sol*
        :rtype: :py:class:`tuple` of :py:class:`float`
        """
        outputs_pred = self.classifier.fit(
            training_data.inputs,
            training_data.outputs
        ).predict(test_data.inputs)
        return (self.__class__._score(test_data.outputs, outputs_pred), )

    def _evaluate_mccv(
        self,
        training_data: Dataset
    ) -> Tuple[float, ...]:
        """Evaluate a solution.

        The *training_data* are split (stratified) into training and test data
        according to
        :py:attr:`~culebra.fitness_function.abc.DatasetScorer.test_prop`
        each time the solution is evalueted and a Monte Carlo cross-validation
        is applied.

        :param training_data: The training dataset
        :type training_data: :py:class:`~culebra.tools.Dataset`
        :return: The fitness values for *sol*
        :rtype: :py:class:`tuple` of :py:class:`float`
        """
        # Split training data into training and test
        training_data, test_data = training_data.split(self.test_prop)
        return self._evaluate_train_test(training_data, test_data)

    def _evaluate_kfcv(
        self,
        training_data: Dataset,
    ) -> Tuple[float, ...]:
        """Evaluate a solution.

        A *k*-fold cross-validation is applied using the *training_data* with
        :py:attr:`~culebra.fitness_function.abc.DatasetScorer.cv_folds`
        folds.

        :param training_data: The training dataset
        :type training_data: :py:class:`~culebra.tools.Dataset`
        :return: The fitness values for *sol*
        :rtype: :py:class:`tuple` of :py:class:`float`
        """
        scores = cross_val_score(
            self.classifier,
            training_data.inputs,
            training_data.outputs,
            cv=StratifiedKFold(n_splits=self.cv_folds),
            scoring=make_scorer(self.__class__._score)
        )
        return (scores.mean(), )


class ClassificationFSScorer(ClassificationScorer):
    """Abstract base fitness function for feature selection on classification problems."""

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
        if sol.features.size > 0:
            return ClassificationScorer.evaluate(
                self, sol, index, representatives
            )
        else:
            return (self._worst_score, )

    def _final_training_test_data(
        self,
        sol: Solution
    ) -> Tuple[Dataset, Dataset]:
        """Get the final training and test data.

        :param sol: Solution to be evaluated. It is used to select the
          features from the datasets
        :type sol: :py:class:`~culebra.abc.Solution`

        :return: The final training and test datasets
        :rtype: :py:class:`tuple` of :py:class:`~culebra.tools.Dataset`
        """
        return (
            self.training_data.select_features(sol.features),
            None if self.test_data is None else
            self.test_data.select_features(sol.features)
        )

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


class RBFSVCScorer(ClassificationScorer):
    """Abstract base class fitness function for RBF SVC optimization."""

    def __init__(
        self,
        training_data: Dataset,
        test_data: Optional[Dataset] = None,
        test_prop: Optional[float] = None,
        cv_folds: Optional[int] = None,
        classifier: Optional[ClassifierMixin] = None
    ) -> None:
        """Construct a fitness function.

        If *test_data* are provided, the whole *training_data* are used to
        train. Otherwise, if *test_prop* is provided, *training_data* are
        split (stratified) into training and test data each time
        :py:meth:`~culebra.FitnessFunction.RBFSVCScorer.evaluate` is
        called and a Monte Carlo cross validation is applied. Finally, if both
        *test_data* and *test_prop* are omitted, a *k*-fold cross-validation
        is applied.

        :param training_data: The training dataset
        :type training_data: :py:class:`~culebra.tools.Dataset`
        :param test_data: The test dataset, defaults to :py:data:`None`
        :type test_data: :py:class:`~culebra.tools.Dataset`, optional
        :param test_prop: A real value in (0, 1) or :py:data:`None`. Defaults
            to :py:data:`None`
        :type test_prop: :py:class:`float`, optional
        :param cv_folds: The number of folds for *k*-fold cross-validation.
            If omitted, :py:attr:`~culebra.fitness_function.DEFAULT_CV_FOLDS`
            is used. Defaults to :py:data:`None`
        :type cv_folds: :py:class:`int`, optional
        :param classifier: The classifier. If set to :py:data:`None`,
            a :py:class:`~sklearn.svm.SVC` with RBF kernels will be used.
            Defaults to :py:data:`None`
        :type classifier: :py:class:`~sklearn.svm.SVC` with
            RBF kernels, optional
        """
        # Init the superclass
        ClassificationScorer.__init__(
            self, training_data, test_data, test_prop, cv_folds, classifier
        )

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
    def classifier(self, value: ClassifierMixin | None) -> None:
        """Set a classifier.

        :param value: The classifier. If set to :py:data:`None`,
            :py:class:`~sklearn.svm.SVC with RBF kernels is chosen
        :type: Any subclass of :py:class:`~sklearn.svm.SVC` with RBF kernels
        :raises TypeError: If set to a value which is not an
            :py:class:`~sklearn.svm.SVC`
        :raises ValueError: If the classifier has not RBF kernels
        """
        if value is not None:
            value = check_instance(
                value, "classifier", cls=SVC
            )

            if value.kernel != 'rbf':
                raise ValueError(
                    f"The classifier has not RBF kernels: {value}"
                )
        else:
            value = SVC(kernel='rbf')

        self._classifier = value

    def evaluate(
        self,
        sol: Solution,
        index: Optional[int] = None,
        representatives: Optional[Sequence[Solution]] = None
    ) -> Tuple[float, ...]:
        """Evaluate a solution.

        :param sol: Solution to be evaluated.
        :type sol:
            :py:class:`~culebra.solution.parameter_optimization.Solution`
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
        # Set the classifier hyperparameters
        hyperparams = sol.values
        self.classifier.C = hyperparams.C
        self.classifier.gamma = hyperparams.gamma

        return ClassificationScorer.evaluate(
            self, sol, index, representatives
        )


class CooperativeFSScorer(FitnessFunction):
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

        return (sol_hyperparams, sol_features)


class CooperativeRBFSVCFSScorer(
    CooperativeFSScorer,
    RBFSVCScorer
):
    """Abstract base class for cooperative FS problems using an RBF SVC classifier."""


# Exported symbols for this module
__all__ = [
    'DatasetScorer',
    'ClassificationScorer',
    'ClassificationFSScorer',
    'RBFSVCScorer',
    'CooperativeFSScorer',
    'CooperativeRBFSVCFSScorer'
]
