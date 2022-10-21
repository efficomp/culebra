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

"""Fitness functions for cooperative problems.

This module provides several fitness functions designed to solve feature
selection problems using a cooperative wrapper approach.
"""

from __future__ import annotations
from typing import Optional, Tuple
from collections.abc import Sequence
from culebra.base import Fitness
from culebra.base import Individual
from ..feature_selection.feature_selection import KappaNumFeats
from ..classifier_optimization.rbf_svc_optimization import C

__author__ = 'Jesús González'
__copyright__ = 'Copyright 2021, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.1.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


class KappaNumFeatsC(KappaNumFeats, C):
    """Tri-objective fitness class for feature selection.

    Maximizes the Kohen's Kappa index and minimizes the number of features
    that an individual has selected and also de C regularazation
    hyperparameter of the SVM-based classifier used within the wrapper.

    More information about this fitness function can be found in
    [Gonzalez2021]_.
    """

    class Fitness(Fitness):
        """Fitness returned by this fitness function."""

        weights = KappaNumFeats.Fitness.weights + C.Fitness.weights
        """Maximizes the Kohen's Kappa index and minimizes the number of
        features that an individual has selected and also de C regularization
        hyperparameter.
        """

        names = KappaNumFeats.Fitness.names + C.Fitness.names
        """Name of the objectives."""

        thresholds = KappaNumFeats.Fitness.thresholds + C.Fitness.thresholds
        """Similarity threshold for fitness comparisons."""

    # Copy the :py:class:`culebra.fitness_function.classifier_optimizacion.C`
    # properties
    __init__ = C.__init__
    classifier = C.classifier

    def evaluate(
        self,
        ind: Individual,
        index: Optional[int] = None,
        representatives: Optional[Sequence[Individual]] = None
    ) -> Tuple[float, ...]:
        """Evaluate an individual.

           This fitness function assumes that:

             * *representatives[0]*: Codes the SVC hyperparameters
               (C and gamma). Thus, it is an instance of
               :py:class:`genotype.classifier_optimization.Individual`
             * *representatives[1:]*: The remaining individuals code the
               features selected, each individual a different range of
               features. All of them are instances of
               :py:class:`genotype.feature_selection.Individual`

        :param ind: Individual to be evaluated.
        :type ind: :py:class:`~base.Individual`
        :param index: Index where *ind* should be inserted in the
            representatives sequence to form a complete solution for the
            problem
        :type index: :py:class:`int`
        :param representatives: Representative individuals of each species
            being optimized
        :type representatives: :py:class:`~collections.abc.Sequence` of
            :py:class:`~base.Individual`, ignored
        :return: The fitness of *ind*
        :rtype: :py:class:`tuple` of py:class:`float`
        """
        # Number of representatives
        num_representatives = len(representatives)

        # Hyperparameters individual
        ind_hyperparams = ind if index == 0 else representatives[0]

        # Prototype individual for the final individual containing all the
        # features
        prototype_ind_features = ind if index == 1 else representatives[1]

        # All the features
        all_the_features = []

        # Number of features
        number_features = prototype_ind_features.species.num_feats

        # Update the features and feature min and max indices
        for repr_index in range(1, num_representatives):
            # Choose thge correct individual
            the_ind = (
                ind if repr_index == index else representatives[repr_index]
            )
            # Get the features
            all_the_features += list(the_ind.features)

        # Features individual class
        ind_features_cls = prototype_ind_features.__class__

        # Features individual species class
        ind_features_species_cls = prototype_ind_features.species.__class__

        # Features individual species
        ind_features_species = ind_features_species_cls(number_features)

        # Features individual
        ind_features = ind_features_cls(
            species=ind_features_species,
            fitness_cls=self.Fitness,
            features=all_the_features
        )

        # Set the classifier hyperparameters
        self.classifier.C = ind_hyperparams.values.C
        self.classifier.gamma = ind_hyperparams.values.gamma

        return (
            KappaNumFeats.evaluate(self, ind_features) +
            C.evaluate(self, ind_hyperparams)
        )


# Exported symbols for this module
__all__ = ['KappaNumFeatsC']
