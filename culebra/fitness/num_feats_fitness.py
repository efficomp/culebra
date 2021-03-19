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

"""Provides the :py:class:`~fitness.num_feats_fitness.NumFeatsFitness`
class.
"""

from culebra.base.fitness import Fitness

__author__ = 'Jesús González'
__copyright__ = 'Copyright 2021, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.1.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


class NumFeatsFitness(Fitness):
    """Dummy single-objective :py:class:`~base.fitness.Fitness` subclass for
    testing purposes.

    Minimize the number of features that an individual has selected. Do not
    take into account any classification index.
    """

    weights = (-1.0,)
    """Minimize the number of features that an individual has selected."""

    names = ("NF",)
    """Name of the objectives."""

    def __init__(self, **params):
        """Create the fitness.

        :param thresholds: Thresholds to assume if two fitness values are
            equivalent. If only a single value is provided, the same threshold
            will be used for all the objectives. A different threshold can be
            provided for each objective with a
            :py:class:`~collections.abc.Sequence`. Defaults to
            :py:attr:`~base.fitness.DEFAULT_THRESHOLD`
        :type thresholds: :py:class:`float` or a
            :py:class:`~collections.abc.Sequence` of :py:class:`float` numbers,
            optional
        :raises TypeError: If *thresholds* is not a :py:class:`float` value or
            a :py:class:`~collections.abc.Sequence` of :py:class:`float`
            numbers
        :raises ValueError: If a negative threshold is provided
        """
        super().__init__(**params)

    def eval(self, ind, dataset):
        """Evaluate an individual.

        Returns the number of features selected by *ind*.

        :param ind: The individual
        :type ind: Any subclass of :py:class:`~base.individual.Individual`
        :param dataset: A dataset
        :type dataset: :py:class:`~base.dataset.Dataset`
        :return: The number of features selected by *ind*
        :rtype: :py:class:`tuple`
        """
        return (ind.num_feats,)


# Perform some tests
if __name__ == '__main__':
    from culebra.base.species import Species
    from culebra.individual.int_vector import IntVector

    NUM_FEATS = 10

    s = Species(NUM_FEATS)
    fitness = NumFeatsFitness()
    ind = IntVector(s, fitness)
    ind.fitness.setValues(fitness.eval(ind, None))
    print(f"Individual: {ind}")
    print(f"{fitness.names}: {ind.fitness}")
