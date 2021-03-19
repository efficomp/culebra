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

"""Provide some feature metrics."""

import numpy as np
import pandas as pd

__author__ = 'Jesús González'
__copyright__ = 'Copyright 2021, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.1.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


def relevance(individuals):
    """Return the relevance of the features selected by a wrapper method.

    :param individuals: Best individuals returned by the wrapper method.
    :type individuals: Any iterable type.
    :return: The relevance of each feature appearing in the individuals.
    :rtype: :py:class:`~pandas.Series`
    """
    # species of the individuals
    s = individuals[0].species

    # all relevances are initialized to 0
    relevances = dict((feat, 0) for feat in np.arange(s.min_feat,
                                                      s.max_feat + 1))
    n_ind = 0
    for ind in individuals:
        n_ind += 1
        for feat in ind.features:
            if feat in relevances:
                relevances[feat] += 1
            else:
                relevances[feat] = 1

    relevances = {feat: relevances[feat] / n_ind for feat in relevances}

    # return pd.Series(relevances).sort_values(ascending=False)
    return pd.Series(relevances).sort_index()


def rank(individuals):
    """Return the rank of the features selected by a wrapper method.

    :param individuals: Best individuals returned by the wrapper method.
    :type individuals: Any iterable type.
    :return: The relevance of each feature appearing in the individuals.
    :rtype: :py:class:`~pandas.Series`
    """
    # Obtain the relevance of each feature. The series is sorted, starting with
    # the most relevant feature
    relevances = relevance(individuals)

    # Obtain the different relevance values
    rel_values = np.sort(np.unique(relevances.values))[::-1]

    ranks = {}

    index = 0
    for val in rel_values:
        feats = [feat for feat, rel in relevances.items() if rel == val]
        n_feats = len(feats)
        rank = (2*index + n_feats - 1) / 2
        index += n_feats
        for feat in feats:
            ranks[feat] = rank

    return pd.Series(ranks).sort_index()
