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

"""Cooperative feature selection fitness functions.

This sub-module provides fitness functions designed to the cooperative solving
of a feature selection problem while the classifier hyperparamters are also
being optimized. It provides the following fitness functions:

  * :py:class:`~culebra.fitness_function.cooperative.FSSVCScorer`:
    Abstract base class for all the fitness functions of cooperative FS
    problems.
"""

from __future__ import annotations

from typing import Optional, Tuple
from collections.abc import Sequence

from sklearn.svm import SVC

from culebra.abc import Solution, Fitness

from culebra.fitness_function.feature_selection.abc import (
    FSScorer,
    FSDatasetScorer,
    FSClassificationScorer
)
from culebra.fitness_function.feature_selection import (
    FSMultiObjectiveDatasetScorer
)
from culebra.fitness_function.svc_optimization.abc import (
    SVCScorer,
    RBFSVCScorer
)


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2025, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.4.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


class FSSVCScorer(FSMultiObjectiveDatasetScorer):
    """Abstract base class fitness function for cooperative FS problems."""

    def __init__(
        self,
        fs_classification_scorer,
        *remaining_objectives
    ) -> None:
        """Construct a cooperative multi-objective fitness function.

        All the objectives that analyze a dataset *must* be
        :py:class:`~culebra.fitness_function.feature_selection.abc.FSClassificationScorer`
        instances using an
        :py:class:`~sklearn.svm.SVC` with RBF kernels. That is, no
        :py:class:`~culebra.fitness_function.svc_optimization.abc.RBFSVCScorer`
        objectives are allowed.

        :param fs_classification_scorer: A classification FS objective
            responsible of the
            :py:attr:`~culebra.fitness_function.cooperative.FSSVCScorer.num_nodes`
            property and the
            :py:meth:`~culebra.fitness_function.cooperative.FSSVCScorer.heuristic`
            implementations
        :type fs_classification_scorer:
            :py:class:`~culebra.fitness_function.feature_selection.abc.FSClassificationScorer`

        :param remaining_objectives: Remaining objectives for this fitness
            function
        :type remaining_objectives:
            :py:class:`~culebra.fitness_function.abc.SingleObjectiveFitnessFunction`
        """
        # Check the objectives
        for obj_idx, obj in enumerate(
            (fs_classification_scorer, *remaining_objectives)
        ):
            if isinstance(obj, FSScorer):
                if isinstance(obj, FSClassificationScorer):
                    if not isinstance(obj.classifier, SVC):
                        raise ValueError(
                            f"Objective {obj_idx} must use an SVC"
                        )
                    elif obj.classifier.kernel != 'rbf':
                        raise ValueError(
                            f"Objective {obj_idx} must use an RBF kernel"
                        )
                elif isinstance(obj, FSDatasetScorer):
                    raise ValueError(
                            f"Objective {obj_idx} is not allowed"
                        )
            elif isinstance(obj, SVCScorer):
                if isinstance(obj, RBFSVCScorer):
                    raise ValueError(
                            f"Objective {obj_idx} is not allowed"
                        )

        super().__init__(fs_classification_scorer, *remaining_objectives)

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
            fitness_cls=self.fitness_cls,
            features=all_the_features
        )

        return (sol_hyperparams, sol_features)

    def evaluate(
        self,
        sol: Solution,
        index: Optional[int] = None,
        representatives: Optional[Sequence[Solution]] = None
    ) -> Fitness:
        """Evaluate a solution.

           It is assumed that:

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
        :type index: :py:class:`int`, optional
        :param representatives: Representative solutions of each species
            being optimized
        :type representatives: A :py:class:`~collections.abc.Sequence`
            containing instances of :py:class:`~culebra.abc.Solution`,
            optional
        :return: The fitness for *sol*
        :rtype: :py:class:`~culebra.abc.Fitness`
        """
        # Assemble the solution and representatives to construct a complete
        # solution for each of the problems solved cooperatively
        (sol_hyperparams, sol_features) = self.construct_solutions(
            sol, index, representatives
        )

        for obj_idx, obj in enumerate(self.objectives):
            if isinstance(obj, FSScorer):
                if isinstance(obj, FSClassificationScorer):
                    obj.classifier.C = sol_hyperparams.values.C
                    obj.classifier.gamma = sol_hyperparams.values.gamma

                obj.evaluate(sol_features)
                sol.fitness.update_value(
                    sol_features.fitness.values[obj_idx], obj_idx
                )
            elif isinstance(obj, SVCScorer):
                obj.evaluate(sol_hyperparams)
                sol.fitness.update_value(
                    sol_hyperparams.fitness.values[obj_idx], obj_idx
                )

        return sol.fitness


# Exported symbols for this module
__all__ = [
    'FSSVCScorer'
]
