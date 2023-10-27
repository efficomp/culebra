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

"""Cooperative feature selection fitness functions.

This sub-module provides fitness functions designed to the cooperative solving
of a feature selection problem while the classifier hyperparamters are also
being optimized. It provides the following fitness functions:

  * :py:class:`~culebra.fitness_function.cooperative.KappaNumFeatsC`:
    Tri-objective fitness class for feature selection. Maximizes the Kohen's
    Kappa index and minimizes the number of features that a solution has
    selected and also de C regularazation hyperparameter of the SVM-based
    classifier.

"""

from __future__ import annotations

from typing import Optional, Tuple
from collections.abc import Sequence

from culebra.abc import Fitness, Solution

from .feature_selection import KappaNumFeats
from .svc_optimization import C


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2023, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.3.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


class KappaNumFeatsC(KappaNumFeats, C):
    """Tri-objective fitness class for feature selection.

    Maximizes the Kohen's Kappa index and minimizes the number of features
    that a solution has selected and also de C regularazation
    hyperparameter of the SVM-based classifier.

    More information about this fitness function can be found in
    [Gonzalez2021]_.

    .. [Gonzalez2021] J. González, J. Ortega, J. J. Escobar, M. Damas.
       *A lexicographic cooperative co-evolutionary approach for feature
       selection*. **Neurocomputing**, 463:59-76, 2021.
       https://doi.org/10.1016/j.neucom.2021.08.003.
    """

    class Fitness(Fitness):
        """Fitness returned by this fitness function."""

        weights = KappaNumFeats.Fitness.weights + C.Fitness.weights
        """Maximizes the Kohen's Kappa index and minimizes the number of
        features that a solution has selected and also de C regularization
        hyperparameter.
        """

        names = KappaNumFeats.Fitness.names + C.Fitness.names
        """Name of the objectives."""

        thresholds = KappaNumFeats.Fitness.thresholds + C.Fitness.thresholds
        """Similarity threshold for fitness comparisons."""

    # Copy the :py:class:`culebra.fitness_function.parameter_optimizacion.C`
    # properties
    __init__ = C.__init__
    classifier = C.classifier

    def evaluate(
        self,
        sol: Solution,
        index: Optional[int] = None,
        representatives: Optional[Sequence[Solution]] = None
    ) -> Tuple[float, ...]:
        """Evaluate a solution.

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
        :return: The fitness of *sol*
        :rtype: :py:class:`tuple` of py:class:`float`
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

        # Set the classifier hyperparameters
        self.classifier.C = sol_hyperparams.values.C
        self.classifier.gamma = sol_hyperparams.values.gamma

        return (
            KappaNumFeats.evaluate(self, sol_features) +
            C.evaluate(self, sol_hyperparams)
        )


# Exported symbols for this module
__all__ = ['KappaNumFeatsC']
