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

"""Provides the :py:class:`~fitness.kappa_index_fitness.KappaIndexFitness`
class.
"""
import numbers
from culebra.base.fitness import Fitness
from sklearn.base import ClassifierMixin
from sklearn.metrics import cohen_kappa_score

__author__ = 'Jesús González'
__copyright__ = 'Copyright 2021, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.1.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


class KappaIndexFitness(Fitness):
    """Single-objective fitness class for classification problems.

    Maximizes the Kohen's Kappa index.
    """

    weights = (1.0,)
    """Maximizes the valiation Kappa index."""

    names = ("Kappa",)
    """Name of the objectives."""

    def __init__(self, **params):
        """Create the fitness.

        If :py:attr:`valid_prop` is not `None`, the classifier is trained with
        (1 - :py:attr:`valid_prop`) proportion of the samples and validated
        with the remaining samples. Otherwise the classifier is trained and
        tested with all the available data.

        :param valid_prop: Validation proportion. If not `None`, the
            classifier is trained with (1 - *valid_prop*) proportion of the
            samples and validated with the remaining samples. Otherwise the
            classifier is trained and tested with all the available data.
            Defaults to `None`.
        :type valid_prop: A real number, optional
        :param classifier: The classifier to be used within this wrapper.
        :type classifier: Any subclass of
            :py:class:`~sklearn.base.ClassifierMixin`
        :param thresholds: Thresholds to assume if two fitness values are
            equivalent. If only a single value is provided, the same threshold
            will be used for all the objectives. A different threshold can be
            provided for each objective with a
            :py:class:`~collections.abc.Sequence`. Defaults to
            :py:attr:`~base.fitness.DEFAULT_THRESHOLD`
        :type thresholds: :py:class:`float` or a
            :py:class:`~collections.abc.Sequence` of :py:class:`float` numbers,
            optional
        :raises TypeError: If *valid_prop* is not a real number
        :raises ValueError: If *valid_prop* is not in (0, 1)
        :raises TypeError: If *classifier* is not a valid classifier
        :raises TypeError: If *thresholds* is not a :py:class:`float` value or
            a :py:class:`~collections.abc.Sequence` of :py:class:`float`
            numbers
        :raises ValueError: If a negative threshold is provided
        """
        super().__init__(**params)

        # Get the validation poroportion
        self.valid_prop = params.pop('valid_prop', None)

        # Get the classifer
        self.classifier = params.pop('classifier', None)

    def eval(self, ind, dataset):
        """Evaluate an individual.

        :param ind: The individual
        :type ind: Any subclass of :py:class:`~base.individual.Individual`
        :param dataset: A dataset
        :type dataset: :py:class:`~base.dataset.Dataset`
        :return: The Kappa index achieved by the classifier with the validation
            data when using the features selected by *ind*
        :rtype: :py:class:`tuple`
        """
        kappa = 0
        if ind.num_feats > 0:
            if self.valid_prop is not None:
                # Split data into training and validation
                training, validation = dataset.split(self.valid_prop)
            else:
                # Use all the data
                training = validation = dataset

            y_pred = self.classifier.fit(
                training.X[:, ind.features],
                training.y).predict(validation.X[:, ind.features])
            kappa = cohen_kappa_score(validation.y, y_pred)

        return (kappa,)

    @property
    def valid_prop(self):
        """Proportion of data used for validation.

        :getter: Return the proportion
        :setter: Set a new value for the porportion. A real value in (0, 1) or
            `None` is expected
        :type: :py:class:`float`
        :raises TypeError: If set with a value which is not a real number
        :raises ValueError: If set with a value which is not in (0, 1)
        """
        return self.__valid_prop

    @valid_prop.setter
    def valid_prop(self, value):
        """Set a value for proportion of data used for validation.

        :param value: A real value in (0, 1) or `None`
        :type value: A real number or `None`
        :raises TypeError: If *value* is not a real number
        :raises ValueError: If *value* is not in (0, 1)
        """
        if value is not None:
            if not isinstance(value, numbers.Real):
                raise TypeError("valid_prop must be a real number")
            if not 0 < value < 1:
                raise ValueError("valid_prop must be in (0, 1)")

        self.__valid_prop = value

    @property
    def classifier(self):
        """Classifier applied within this wrapper.

        :getter: Return the classifier
        :setter: Set a new classifier
        :type: Any subclass of :py:class:`~sklearn.base.ClassifierMixin`
        :raises TypeError: If set to a value which is not a classifier
        """
        return self.__classifier

    @classifier.setter
    def classifier(self, value):
        """Set a classifier.

        :param value: The classifier
        :type value: Any subclass of :py:class:`~sklearn.base.ClassifierMixin`
        :raises TypeError: If *value* is not a classifier
        """
        if value is not None:
            if not isinstance(value, ClassifierMixin):
                raise TypeError("Not a valid classifier")

        self.__classifier = value


# Perform some tests
if __name__ == '__main__':

    from sklearn.naive_bayes import GaussianNB
    from culebra.base.species import Species
    from culebra.base.dataset import Dataset
    from culebra.individual.int_vector import IntVector as Individual

    # Dataset
    DATASET_PATH = ('https://archive.ics.uci.edu/ml/'
                    'machine-learning-databases/statlog/australian/'
                    'australian.dat')

    # Proportion of training data used for validation
    VALID_PROP = 0.25

    # Minimum individual size
    MIN_SIZE = 1

    dataset = Dataset.load(DATASET_PATH, output_index=-1)

    # Normalize inputs between 0 and 1
    dataset.normalize()

    fitness = KappaIndexFitness(valid_prop=VALID_PROP, classifier=GaussianNB())

    # Create the species for the individuals of the population
    species = Species(num_feats=dataset.num_feats, min_size=MIN_SIZE)
    ind = Individual(species, fitness)

    ind.fitness.setValues(fitness.eval(ind, dataset))
    print(ind, ind.fitness)

    fitness.valid_prop = None
    ind.fitness.setValues(fitness.eval(ind, dataset))
    print(ind, ind.fitness)
