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

"""Provide some metrics about the selected features obtained by a wrapper."""

from __future__ import annotations
from collections.abc import Sequence
import numpy as np
from pandas import Series
from culebra.base import Base
from . import Individual

__author__ = "Jesús González"
__copyright__ = "Copyright 2021, EFFICOMP"
__license__ = "GNU GPL-3.0-or-later"
__version__ = "0.1.1"
__maintainer__ = "Jesús González"
__email__ = "jesusgonzalez@ugr.es"
__status__ = "Development"


class Metrics(Base):
    """Provide some metrics about the selected features obtained by a wrapper.

    Evaluate the set of solutions found by a :py:class:`~base.Wrapper` and
    calculate some metrics about the frequency of each selected feature. More
    information about such metrics can be found in:

    .. [Gonzalez2019] J. González, J. Ortega, M. Damas, P. Martín-Smith,
       John Q. Gan. *A new multi-objective wrapper method for feature
       selection - Accuracy and stability analysis for BCI*.
       **Neurocomputing**, 333:407-418, 2019.
       https://doi.org/10.1016/j.neucom.2019.01.017.
    """

    @staticmethod
    def relevance(individuals: Sequence[Individual]) -> Series:
        """Return the relevance of the features selected by a wrapper method.

        The relevance is calculated according to the method proposed in
        [Gonzalez2019]_

        :param individuals: Best individuals returned by the wrapper method.
        :type individuals: :py:class:`~collections.abc.Sequence`
        :return: The relevance of each feature appearing in the individuals.
        :rtype: :py:class:`~pandas.Series`
        """
        # species of the individuals
        species = individuals[0].species

        # all relevances are initialized to 0
        relevances = dict(
            (feat, 0) for feat in np.arange(
                species.min_feat, species.max_feat + 1)
        )
        n_ind = 0
        for ind in individuals:
            n_ind += 1
            for feat in ind.features:
                if feat in relevances:
                    relevances[feat] += 1
                else:
                    relevances[feat] = 1

        relevances = {feat: relevances[feat] / n_ind for feat in relevances}

        return Series(relevances).sort_index()

    @staticmethod
    def rank(individuals: Sequence[Individual]) -> Series:
        """Return the rank of the features selected by a wrapper method.

        The rank is calculated according to the method proposed in
        [Gonzalez2019]_

        :param individuals: Best individuals returned by the wrapper method.
        :type individuals: :py:class:`~collections.abc.Sequence`
        :return: The relevance of each feature appearing in the individuals.
        :rtype: :py:class:`~pandas.Series`
        """
        # Obtain the relevance of each feature. The series is sorted, starting
        # with the most relevant feature
        relevances = Metrics.relevance(individuals)

        # Obtain the different relevance values
        rel_values = np.sort(np.unique(relevances.values))[::-1]

        ranks = {}

        index = 0
        for val in rel_values:
            feats = [feat for feat, rel in relevances.items() if rel == val]
            n_feats = len(feats)
            the_rank = (2 * index + n_feats - 1) / 2
            index += n_feats
            for feat in feats:
                ranks[feat] = the_rank

        return Series(ranks).sort_index()


# Exported symbols for this module
__all__ = ["Metrics"]
