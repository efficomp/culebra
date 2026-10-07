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
# This work is supported by projects PGC2018-098813-B-C31 and
# PID2022-137461NB-C31, both funded by the Spanish "Ministerio de Ciencia,
# Innovación y Universidades" and by the European Regional Development Fund
# (ERDF).

"""Implementation of some heuristics for ACO algorithms."""

import warnings

import numpy as np
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer

from culebra.solution.feature_selection import Species, IntSolution
from culebra.fitness_func.abc import SingleObjectiveFitnessFunction
from culebra.fitness_func.tsp.abc import TSPFitnessFunction
from culebra.fitness_func.feature_selection.abc import FSClassificationScorer
from culebra.fitness_func.feature_selection import Accuracy


def tsp_default_heuristic(
        training_fitness_func: TSPFitnessFunction
) -> tuple[np.ndarray[float], ...]:
    """Compute the default heuristic matrices for a TSP problem

    Arcs from a node to itself have a heuristic value of 0. For the remaining
    arcs, the reciprocal of their nodes distance is used as heuristic.

    :param training_fitness_func: Fitness function used during training.
    :type training_fitness_func:
        ~culebra.fitness_func.tsp.abc.TSPFitnessFunction

    :return: A sequence of heuristic matrices. One for each objective.
    :rtype: tuple[~numpy.ndarray[float]]
    """
    def single_obj_heuristic(
        training_fitness_func: TSPFitnessFunction
    ) -> tuple[np.ndarray[float]]:
        """Compute the heuristic of a single-objective fitness function.

        :param training_fitness_func: Single-objective fitness function.
        :type training_fitness_func:
            ~culebra.fitness_func.tsp.abc.TSPFitnessFunction

        :return: A sequence with one heuristic matrix.
        :rtype: tuple[~numpy.ndarray[float]]
        """
        distance = training_fitness_func.distance
        with np.errstate(divide='ignore'):
            heur = np.where(
                distance != 0.,
                1 / distance,
                0.
            )

        for node in range(training_fitness_func.num_nodes):
            heur[node][node] = 0

        return (heur,)

    # If the training fitness function is single-objective
    if isinstance(training_fitness_func, SingleObjectiveFitnessFunction):
        return single_obj_heuristic(training_fitness_func)

    # If it is multi-objective
    heuristics = ()
    for obj in training_fitness_func.objectives:
        heuristics += single_obj_heuristic(obj)

    return heuristics


def fs_rough_set_heuristic(
    training_fitness_func: FSClassificationScorer
) -> tuple[np.ndarray[float], ...]:
    r"""Compute a heuristic vector from the training dataset.

    It is computed as proposed in [Ke2010]_. For each feature :math:`c`,
    the heuristic is defined as:

    .. math::

        \eta(c) = \exp\left(\frac{|POS_c(D)|}{|U|}\right)

    where :math:`POS_c(D)` is the positive region induced by feature
    :math:`c`, :math:`D` is the decision attribute, and :math:`U` is
    the universe of samples. The term :math:`|POS_c(D)| / |U|`
    corresponds to the rough-set dependency degree of the feature.

    A sample belongs to the positive region if its equivalence class,
    induced by the feature value, contains samples from a single
    decision class only.

    :param training_fitness_func: Fitness function used during training.
    :type training_fitness_func:
        ~culebra.fitness_func.feature_selection.abc.FSClassificationScorer

    :return: A sequence of heuristic vectors for each feature.
    :rtype: tuple[~numpy.ndarray[float]]

    .. [Ke2010] L. Ke, Z. Feng, Z. Xu, K. Shang, Y. Wang.
        *A multiobjective ACO algorithm for rough feature selection*, in:
        **2010 2nd Pacific-Asia Conference on Circuits, Communications and
        System, PACCS 2010**, Vol. 1, Beijing, China, 2010, pp. 207–210.
        https://doi.org/10.1109/paccs.2010.5627071.
    """
    training_data = training_fitness_func.training_data
    heuristics = np.empty(training_data.num_feats, dtype=float)

    for feat_idx in range(training_data.num_feats):
        feature_values = training_data.inputs[:, feat_idx]

        positive_region_size = 0

        # Equivalence classes induced by the feature
        unique_values, inverse = np.unique(
            feature_values,
            return_inverse=True
        )

        for eq_class in range(len(unique_values)):
            class_outputs = training_data.outputs[inverse == eq_class]

            # Pure equivalence class
            if np.all(class_outputs == class_outputs[0]):
                positive_region_size += class_outputs.size

        dependency = positive_region_size / training_data.size
        heuristics[feat_idx] = np.exp(dependency)

    return (heuristics,)


def fs_accuracy_heuristic(
    training_fitness_func: FSClassificationScorer
) -> tuple[np.ndarray[float], ...]:
    """Compute a heuristic vector from the predictive performance of
    individual features.

    The heuristic value of a feature is defined as the classification
    accuracy obtained when the feature is evaluated as a single-feature
    subset. The same classifier and evaluation settings used by the training
    fitness function are reused to compute the heuristic values.

    :param training_fitness_func: Fitness function used during training.
    :type training_fitness_func:
        ~culebra.fitness_func.feature_selection.abc.FSClassificationScorer

    :return: A sequence of heuristic vectors for each feature.
    :rtype: tuple[~numpy.ndarray[float]]
    """
    # Define an accuracy fitness function with the same parameters as the
    # training fitness function
    heuristic_fitness_func = Accuracy(
        training_data=training_fitness_func.training_data,
        test_data=training_fitness_func.test_data,
        cv_num_folds=training_fitness_func.cv_num_folds,
        cv_fixed_folds=training_fitness_func.cv_fixed_folds,
        classifier=training_fitness_func.classifier

    )
    num_feats = heuristic_fitness_func.training_data.num_feats
    species = Species(num_feats)

    heuristics = np.empty(num_feats, dtype=float)

    # Evaluate each feature independently as a candidate solution
    for feat_idx in range(num_feats):
        sol = IntSolution(
            species,
            heuristic_fitness_func.fitness_cls,
            [feat_idx]
        )
        heuristics[feat_idx] = heuristic_fitness_func.evaluate(sol)[0]

    return (heuristics,)


def fs_pearson_heuristic(
    training_fitness_func: FSClassificationScorer
) -> tuple[np.ndarray[float], ...]:
    r"""Compute a heuristic matrix based on feature correlations.

    It is computed as proposed in [Ghosh2020]_. The heuristic between two
    features :math:`i` and :math:`j` is computed as:

    .. math::

        \eta_{ij} = \frac{1}{1 + |\rho(i, j)|}

    where :math:`\rho(i, j)` is the Pearson correlation coefficient
    between features :math:`i` and :math:`j`.

    Features with low correlation receive larger heuristic values,
    encouraging the construction of subsets with low redundancy.

    :param training_fitness_func: Fitness function used during training.
    :type training_fitness_func: FSClassificationScorer

    :return: A sequence of heuristic matrices.
    :rtype: tuple[~numpy.ndarray[float]]
    
    .. [Ghosh2020] M. Ghosh, R. Guha, R. Sarkar, A. Abraham.
        *A wrapper-filter feature selection technique based on ant colony
        optimization*, **Neural Computing and Applications**, 32(12):7839-7857,
        2020. https://doi.org/10.1007/s00521-019-04171-3.
    """
    # Input values
    inputs = training_fitness_func.training_data.inputs

    # Correlation matrix
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(inputs, rowvar=False)

    # Ensure a 2D matrix for single-feature datasets
    corr = np.atleast_2d(corr)

    # Constant features
    constant_feats = np.isclose(
        np.std(inputs, axis=0),
        0.0
    )

    # Nans to 0
    corr = np.nan_to_num(corr, nan=0.0)

    # Heuristic values
    heuristics = 1.0 / (1.0 + np.abs(corr))

    # Fix heuristic values for constant features
    heuristics[constant_feats, :] = 0.0
    heuristics[:, constant_feats] = 0.0

    # Fix the diagonal
    np.fill_diagonal(heuristics, 0.0)

    return (heuristics,)


def fs_information_theory_heuristic(
    training_fitness_func: FSClassificationScorer
) -> tuple[np.ndarray[float], ...]:
    r"""Compute a heuristic matrix based on information theory.
    
    This heuristic is inspired by the information-theoretic heuristic
    proposed in [Wang2023]_. The original work was developed for a Binary
    Ant Colony Optimization (BACO) algorithm, where each feature can be
    either selected or not selected and the heuristic is defined over
    transitions between such states.
    
    This implementation adapts the original idea to a conventional ACO
    feature-selection algorithm where ants construct a subset by moving
    between features. Since the concept of feature selection states does
    not exist in this setting, a single heuristic matrix is used.
    
    The heuristic between features :math:`i` and :math:`j` is defined as:
    
    .. math::
    
        \eta_{ij}
        =
        SU(X_j, C)
        \left(
            1 - |\rho(X_i, X_j)|
        \right)
    
    where:
    
    * :math:`SU(X_j, C)` is the symmetric uncertainty between feature
      :math:`j` and the class labels, measuring the relevance of the
      feature to the classification task.
    
    * :math:`\rho(X_i, X_j)` is the Pearson correlation coefficient
      between features :math:`i` and :math:`j`, measuring their redundancy.
    
    Therefore, transitions towards features that are strongly related to
    the class labels and weakly correlated with the currently selected
    feature receive larger heuristic values.
    
    Compared with the original BACO formulation, this adaptation:
    
    * Removes the notion of "selected" and "not selected" feature states.
    * Uses a single heuristic matrix instead of state-dependent
      heuristics.
    * Preserves the original relevance-redundancy trade-off proposed in
      [Wang2023]_.
    * Estimates feature relevance using symmetric uncertainty, as in the
      original work.
    * Estimates pairwise redundancy using Pearson correlation instead of
      symmetric uncertainty.
    
    The redundancy term was modified to improve scalability on
    high-dimensional datasets. Computing symmetric uncertainty for every
    pair of features requires evaluating an information-theoretic measure
    for :math:`O(m^2)` feature pairs, which becomes prohibitively expensive
    for datasets containing hundreds or thousands of features. Pearson
    correlation provides a substantially cheaper approximation of feature
    redundancy while preserving the original objective of discouraging the
    selection of highly redundant features.
    
    :param training_fitness_func: Fitness function used during training.
    :type training_fitness_func: FSClassificationScorer
    
    :return: A sequence of heuristic matrices.
    :rtype: tuple[~numpy.ndarray[float], ...]
    
    .. [Wang2023] Z. Wang, S. Gao, M. C. Zhou, S. Sato, J. Cheng, J. Wang.
        *Information-theory-based nondominated sorting ant colony optimization
        for multiobjective feature selection in classification*,
        **IEEE Transactions on Cybernetics** 53(8):5276-5289, 2023.
        DOI: https://doi.org/10.1109/TCYB.2022.3185554
    """
    def discretize(v: np.ndarray) -> np.ndarray:
        """Discretize a variable.
        
        Quantile-based discretization is used to obtain bins containing a
        similar number of samples. The number of bins is limited by the
        number of distinct values present in the variable.
    
        Constant variables are returned unchanged, as they contain a single
        distinct value only.
    
        :param v: Variable to discretize.
        :type v: numpy.ndarray
    
        :return: Ordinal-encoded discrete representation of the variable.
        :rtype: numpy.ndarray
        """
        # Number of different values in v
        num_values = np.unique(v).size
        
        # Maximum number of bins
        max_bins = 10

        # Constant variables do not need discretization
        if num_values == 1:
            return v
    
        # Limit the number of bins while ensuring it does not exceed
        # the number of distinct values.
        num_bins = min(max_bins, num_values)
    
        discretizer = KBinsDiscretizer(
            n_bins=num_bins,
            encode="ordinal",
            strategy="quantile",
            quantile_method='averaged_inverted_cdf'
        )

        # Transform the variable into an ordinal discrete representation.
        # The reshape is required because scikit-learn expects a 2D array
        # with shape (n_samples, n_features).
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Bins whose width are too small.*"
            )

            return discretizer.fit_transform(
                v.reshape(-1, 1)
            ).ravel()

    def symmetric_uncertainty(
        x: np.ndarray,
        y: np.ndarray
    ) -> float:
        """Compute the symmetric uncertainty between two variables.
    
        Symmetric uncertainty is defined as:
    
        .. math::
    
            SU(X, Y)
            =
            \frac{2 I(X;Y)}
            {H(X) + H(Y)}
    
        where :math:`I(X;Y)` is the mutual information between
        :math:`X` and :math:`Y`, and :math:`H(X)` and :math:`H(Y)` are their
        entropies.
    
        The variables are assumed to be discrete. Any required
        discretization must be performed before calling this function.
    
        :param x: First variable.
        :type x: numpy.ndarray
    
        :param y: Second variable.
        :type y: numpy.ndarray
    
        :return: Symmetric uncertainty.
        :rtype: float
        """
        # Constant variables have zero entropy and therefore
        # zero symmetric uncertainty.
        if np.unique(x).size == 1 or np.unique(y).size == 1:
            return 0.0

        # Entropies of x and y.
        # Since I(x;x) = H(x), mutual_info_score can be used to obtain
        # the entropy of a discrete variable.
        h_x = mutual_info_score(x, x)
        h_y = mutual_info_score(y, y)

        # Mutual information between x and y.
        mi = mutual_info_score(x, y)

        # Symmetric uncertainty:
        # SU(x,y) = 2 I(x;y) / (H(x) + H(y))
        su = 2.0 * mi / (h_x + h_y)

        # Numerical protection to ensure the result remains within the
        # theoretical range [0, 1].
        return np.clip(su, 0.0, 1.0)

    # Dataset
    data = training_fitness_func.training_data

    num_feats = data.num_feats
    inputs = data.inputs
    outputs = data.outputs

    # Constant features
    constant_feats = np.isclose(
        np.std(inputs, axis=0),
        0.0
    )

    # Estimate feature redundancy through Pearson correlation
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(inputs, rowvar=False)

    # Ensure a 2D matrix for single-feature datasets
    corr = np.atleast_2d(corr)

    # Replace NaN correlations generated by constant features
    corr = np.nan_to_num(corr, nan=0.0)

    # Redundancy matrix
    redundancy = np.abs(corr)

    # Relevance of each feature with respect to the class
    relevance = np.empty(num_feats, dtype=float)

    for feat_idx in range(num_feats):
        relevance[feat_idx] = symmetric_uncertainty(
            discretize(inputs[:, feat_idx]),
            outputs
        )

    # Heuristic matrix
    #
    # ηij = SU(Xj, C) * (1 - |ρ(Xi, Xj)|)
    #
    heuristics = (
        relevance[np.newaxis, :]
        * (1.0 - redundancy)
    )

    # Remove transitions involving constant features
    heuristics[constant_feats, :] = 0.0
    heuristics[:, constant_feats] = 0.0

    # No self-transitions
    np.fill_diagonal(heuristics, 0.0)

    return (heuristics,)


# Exported symbols for this module
__all__ = [
    'tsp_default_heuristic',
    'fs_rough_set_heuristic',
    'fs_accuracy_heuristic',
    'fs_pearson_heuristic',
    'fs_information_theory_heuristic'
]
