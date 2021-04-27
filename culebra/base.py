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

"""Base classes of culebra.

Culebra is based in some base classes to define the fundamental pieces that are
necessary to solve a feature selection problem. This module defines:

  * A :py:class:`~base.Base` class from which all the classes of culebra
    inherit
  * A :py:class:`~base.Dataset` class containing samples, where the most
    relevant input features need to be selected
  * An :py:class:`~base.Individual` class, which will be used within the
    :py:class:`~base.Wrapper` class to search the best subset of features
  * A :py:class:`~base.Species` class to define the characteristics of the
    individuals
  * A :py:class:`~base.Fitness` class to guide the search towards optimal
    solutions
  * A :py:class:`~base.Wrapper` class to perform the search
  * A :py:class:`~base.Metrics` class to calculate some metrics about the
    frequency of each selected feature in the solutions found by the
    :py:class:`~base.Wrapper`
"""

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence
import numbers
import copy
import random
import numpy as np
from pandas import Series, read_csv, to_numeric
from sklearn.utils.validation import check_random_state
from sklearn.model_selection import train_test_split
from deap.base import Toolbox
from deap.base import Fitness as DeapFitness
from deap.tools import Statistics


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2021, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.1.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


DEFAULT_DATASET_SEP = '\\s+'
"""Default column separator used within dataset files."""

DEFAULT_FITNESS_THRESHOLD = 0
"""Default similarity threshold for fitness objects."""

DEFAULT_WRAPPER_STATS_NAMES = ("N",)
"""Default statistics calculated within a wrapper each time."""

DEFAULT_WRAPPER_OBJECTIVE_STATS = {"Avg": np.mean, "Std": np.std,
                                   "Min": np.min, "Max": np.max}
"""Default statistics calculated for each objective within a wrapper."""

DEFAULT_WRAPPER_CHECKPOINT_FREQ = 10
"""Default checkpointing frequency for wrappers."""

DEFAULT_WRAPPER_CHECKPOINT_FILE = "checkpoint.gz"
"""Default checkpointing file for wrappers."""


class Base:
    """Define the base class for all objects in culebra."""

    def __copy__(self):
        """Shallow copy the object."""
        cls = self.__class__
        result = cls()
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        """Deepcopy the object.

        :param memo: Object attributes
        :type memo: :py:class:`dict`
        :return: A deep copy of the object
        :rtype: The same than the original object
        """
        cls = self.__class__
        result = cls()
        result.__dict__.update(copy.deepcopy(self.__dict__, memo))
        return result

    def __setstate__(self, state):
        """Set the state of the object.

        :param state: The state
        :type state: :py:class:`dict`
        """
        self.__dict__.update(state)

    def __reduce__(self):
        """Reduce the object.

        :return: The reduction
        :rtype: :py:class:`tuple`
        """
        return (self.__class__, (), self.__dict__)

    def __repr__(self):
        """Return the object representation."""
        cls = self.__class__
        cls_name = cls.__name__
        properties = (
                p for p in dir(cls)
                if isinstance(getattr(cls, p), property) and
                not p.startswith('_')
                      )

        msg = cls_name
        sep = "("
        for prop in properties:
            msg += sep + prop + ": " + getattr(self, prop).__str__()
            sep = ", "

        if sep[0] == '(':
            msg += sep
        msg += ")"
        return msg


class Dataset(Base):
    """Dataset handler."""

    def __init__(self):
        """Create an empty dataset."""
        self._inputs = self._outputs = np.arange(0, dtype=float)

    @property
    def num_feats(self):
        """Get the number of features in the dataset.

        :type: :py:class:`int`
        """
        if self.size == 0:
            return 0

        return self._inputs.shape[1]

    @property
    def size(self):
        """Get the number of samples in the dataset.

        :type: :py:class:`int`
        """
        return self._inputs.shape[0]

    @property
    def inputs(self):
        """Get the input data of the dataset.

        :type: :py:class:`numpy.ndarray`
        """
        return self._inputs

    @property
    def outputs(self):
        """Get the output data of the dataset.

        :type: :py:class:`numpy.ndarray`
        """
        return self._outputs

    def normalize(self, test=None):
        """Normalize the dataset between 0 and 1.

        If a test dataset is provided, both are taken into account to calculate
        the minimum and maximum value and then both are normalized.

        :param test: Test dataset or `None`, defaults to `None`
        :type test: :py:class:`~base.Dataset`, optional
        """
        if test is None:
            self._inputs -= self._inputs.min(axis=0)
            self._inputs /= self._inputs.max(axis=0)
        else:
            min_outputs = np.minimum(self._inputs.min(axis=0),
                                     test._inputs.min(axis=0))
            self._inputs -= min_outputs
            test._inputs -= min_outputs

            max_inputs = np.maximum(self._inputs.max(axis=0),
                                    test._inputs.max(axis=0))
            self._inputs /= max_inputs
            test._inputs /= max_inputs

    def append_random_features(self, num_feats, random_seed=None):
        """Return a new dataset with some random features appended.

        :param num_feats: Number of random features to be appended
        :type num_feats: An :py:class:`int` greater than 0
        :param random_seed: Random seed for the random generator, defaults to
            `None`
        :type random_seed: :py:class:`int`, optional
        :raises TypeError: If the number of random features is not an integer
        :raises ValueError: If the number of random features not greater than
            0
        :return: The new dataset
        :rtype: :py:class:`~base.Dataset`
        """
        # Check num_feats
        if not isinstance(num_feats, numbers.Integral):
            raise TypeError("The number of random features should be an "
                            "integer number")
        if num_feats <= 0:
            raise ValueError("The number of random features should be greater "
                             "than 0")

        # Check the random seed
        random_seed = check_random_state(random_seed)

        # Create an empty dataset
        new = self.__class__()

        # Append num_feats random features to the input data
        new._inputs = np.concatenate(
            (self._inputs, random_seed.rand(self.size, num_feats)), axis=1)

        # Copy the output data
        new._outputs = np.copy(self._outputs)

        # Return the new dataset
        return new

    def split(self, test_prop, random_seed=None):
        """Split the dataset.

        :param test_prop: Proportion of the dataset used as test data.
            The remaining samples will be returned as training data
        :type test_prop: :py:class:`float`
        :param random_seed: Random seed for the random generator, defaults to
            `None`
        :type random_seed: :py:class:`int`, optional
        :raises TypeError: If *test_prop* is not `None` or :py:class:`float`
        :raises ValueError: If *test_prop* is not in (0, 1)
        :return: The training and test datasets
        :rtype: :py:class:`tuple` of :py:class:`~base.Dataset`
        """
        # Check test_prop
        if not isinstance(test_prop, float):
            raise TypeError("The test proportion must be a float value")

        # Check the output_index value
        if not 0 < test_prop < 1:
            raise ValueError("The test proportion must be in (0, 1)")

        # Check the random seed
        random_seed = check_random_state(random_seed)

        (training_inputs,
         test_inputs,
         training_outputs,
         test_outputs) = train_test_split(
             self._inputs, self._outputs, test_size=test_prop,
             stratify=self._outputs, random_state=random_seed)
        training = self.__class__()
        training._inputs = training_inputs
        training._outputs = training_outputs
        test = self.__class__()
        test._inputs = test_inputs
        test._outputs = test_outputs

        return training, test

    @classmethod
    def load(cls, *files, output_index=None, sep=DEFAULT_DATASET_SEP):
        """Load a dataset.

        Datasets can be organized in only one file or in two files. If one
        file per dataset is used, then *output_index* must be used to indicate
        which column stores the output values. If *output_index* is set to
        `None` (its default value), it will be assumed that each dataset is
        composed by two consecutive files, the first one containing the input
        columns and the second one storing the output column. Only the first
        column of the second file will be loaded (just one output value per
        sample).

        :param files: Files containing the dataset. If *output_index* is
            `None`, two files are necessary, the first one containing the input
            columns and the second one containing the output column. Otherwise,
            only one file will be used to access to the whole dataset (input
            and output columns)
        :type files: :py:class:`tuple` of :py:class:`str`, path object or
            file-like objects
        :param output_index: If the dataset is provided with only one file,
            this parameter indicates which column in the file does contain the
            output values. Otherwise this parameter must be set to `None` to
            express that inputs and ouputs are stored in two different files
            Its default value is `None`
        :type output_index: :py:class:`int`, optional
        :param sep: Column separator used within the files. Defaults to
            :py:attr:`~base.DEFAULT_DATASET_SEP`
        :type sep: :py:class:`str`, optional
        :raises TypeError: If at least one dataset is not provided
        :raises TypeError: If *output_index* is not `None` or :py:class:`int`
        :raises TypeError: If *sep* is not a string
        :raises IndexError: If *output_index* is out of range
        :raises RuntimeError: If any input value is not numeric or there is any
            missing data
        :raises RuntimeError: When loading a dataset composed of two files, if
            the file containing the input columns and the file containing the
            output column do not have the same number of rows.
        :return: The dataset
        :rtype: :py:class:`~base.Dataset`
        """
        # If inputs and output data are in separate files
        if output_index is None:
            # Load the training data
            data_x, data_y = Dataset.__load_split_dataset(*files, sep=sep)
        # If inputs and output data are in the same file
        else:
            # Load the training data
            (data_x, data_y) = Dataset.__load_mixed_dataset(
                files[0], output_index=output_index, sep=sep)

        # Replace output labels by int identifiers (only if output is not
        # numeric)
        data_y = Dataset.__labels_to_numeric(data_y)

        # Create an empty dataset
        dataset = cls()

        # Convert data to numpy ndarrays
        dataset._inputs = data_x.to_numpy(dtype=float)
        dataset._outputs = data_y.to_numpy()

        # Return the dataset
        return dataset

    @classmethod
    def load_train_test(cls, *files, test_prop=None, output_index=None,
                        sep=DEFAULT_DATASET_SEP, normalize=False,
                        random_feats=None, random_seed=None):
        """Load the training and test data.

        Datasets can be organized in only one file or in two files. If one
        file per dataset is used, then *output_index* must be used to indicate
        which column stores the output values. If *output_index* is set to
        `None` (its default value), it will be assumed that each dataset is
        composed by two consecutive files, the first one containing the input
        columns and the second one storing the output column. Only the first
        column of the second file will be loaded (just one output value per
        sample).

        If *test_prop* is set a valid proportion, the training and test
        datasets will be generated splitting the first dataset provided.
        Otherwise a test dataset can be provided appending more files, but if
        it isn't, the training dataset is assumed to be also the test dataset.

        :param files: Files containing the data. If *output_index* is `None`,
            two consecutive files are required for each dataset, otherwise,
            each file contains a whole dataset (input and output columns).
        :type files: :py:class:`tuple` of :py:class:`str`, path object or
            file-like objects
        :param test_prop: Proportion of the first dataset used as test data.
            The remaining samples will be used as training data.
        :type test_prop: :py:class:`float`, optional
        :param output_index: If datasets are provided with only one file,
            this parameter indicates which column in the file does contain the
            output values. Otherwise this parameter must be set to `None` to
            express that inputs and ouputs are stored in two different files.
            Its default value is `None`
        :type output_index: :py:class:`int`, optional
        :param sep: Column separator used within the files. Defaults to
            :py:attr:`~base.DEFAULT_DATASET_SEP`
        :type sep: :py:class:`str`, optional
        :param normalize: It `True`, datasets will be normalized into [0, 1].
            Defaults to `False`
        :type normalize: :py:class:`bool`
        :param random_feats: Number of random features to be appended, defaults
            to `None`
        :type random_feats: An :py:class:`int` greater than 0, optional
        :param random_seed: Random seed for the random generator, defaults to
            `None`
        :type random_seed: :py:class:`int`, optional
        :raises TypeError: If at least one dataset is not provided
        :raises TypeError: If *output_index* is not None or :py:class:`int`
        :raises TypeError: If *test_prop* is not None or :py:class:`float`
        :raises TypeError: If *sep* is not a string
        :raises ValueError: If *test_prop* is not in (0, 1)
        :raises IndexError: If *output_index* is out of range
        :raises RuntimeError: If any input value is not numeric or there is any
            missing data
        :raises RuntimeError: When loading a dataset composed of two files, if
            the file containing the input columns and the file containing the
            output column do not have the same number of rows.
        :raises RuntimeError: When training and test datasets do not have the
            same number of columns.
        :return: A :py:class:`tuple` of :py:class:`~base.Dataset`
            containing the training and test datasets
        :rtype: :py:class:`tuple`
        """
        # If two files are used per dataset
        files_per_dataset = 2
        if output_index is not None:
            files_per_dataset = 1

        # Load the training dataset
        training = cls.load(*files, output_index=output_index, sep=sep)

        # No test dataset by the moment...
        test = None

        # If the dataset must be split...
        if test_prop is not None:
            # Split the dataset
            training, test = training.split(test_prop, random_seed)

        # Else, if there are test data files...
        elif len(files) >= 2 * files_per_dataset:
            # Load the test dataset
            test = cls.load(*files[files_per_dataset:],
                            output_index=output_index, sep=sep)

            # Check if training and test data have the same number of columns
            if training.num_feats != test.num_feats:
                raise RuntimeError("Training and test data do not have the "
                                   "same number of features")
        if normalize:
            training.normalize(test)

        if random_feats is not None:
            training = training.append_random_features(random_feats,
                                                       random_seed)
            if test is not None:
                test = test.append_random_features(random_feats, random_seed)

        # The test dataset will be the same dataset by default
        if test is None:
            test = training

        return training, test

    @staticmethod
    def __is_numeric(dataframe):
        """Check if all the elements in a dataframe are numeric.

        :param dataframe: A dataframe
        :type dataframe: :py:class:`~pandas.DataFrame`
        :return: `True` if all the elements are numeric
        :rtype: :py:class:`bool`
        """
        return dataframe.apply(lambda s:
                               to_numeric(s, errors='coerce').notnull().all()
                               ).all()

    @staticmethod
    def __has_missing_data(dataframe):
        """Check if a dataframe has missing values.

        :param dataframe: A dataframe
        :type dataframe: :py:class:`~pandas.DataFrame`
        :return: `True` if there is any missing value
        :rtype: :py:class:`bool`
        """
        return dataframe.isna().values.any()

    @staticmethod
    def __labels_to_numeric(data_s):
        """Replace output labels by numeric identifiers.

        The replacement is performed only if output data is not numeric.

        :param data_s: The outputs
        :type data_s: :py:class:`~pandas.Series`
        :return: A series of numeric values
        :rtype: :py:class:`~pandas.Series`
        """
        values = data_s.to_numpy()

        # If outputs are numeric that's all!
        if np.issubdtype(values.dtype, np.number):
            return data_s

        labels = np.unique(values)
        rep = dict(zip(labels, range(len(labels))))
        return data_s.replace(rep)

    @staticmethod
    def __split_input_output(data, output_index):
        """Split a dataframe into input and output data.

        :param data: A dataframe containing input and output data
        :type data: :py:class:`~pandas.DataFrame`
        :param output_index: Index of the column containing the outuput data
        :type output_index: :py:class:`int`
        :raises TypeError: If *output_index* is not an integer value
        :raises IndexError: If *output_index* is out of range
        :return: A tuple containing a :py:class:`~pandas.DataFrame` with the
            input columns and a :py:class:`~pandas.Series` with the output
            column
        :rtype: :py:class:`tuple`
        """
        # Check the type of output_index
        if not isinstance(output_index, int):
            raise TypeError("The output index must be an integer value")

        # Check the output_index value
        if not -len(data.columns) <= output_index < len(data.columns):
            raise IndexError("Output index out of range")

        output_s = data.iloc[:, output_index]
        inputs_df = data
        inputs_df.drop(inputs_df.columns[output_index], axis=1, inplace=True)

        return inputs_df, output_s

    @staticmethod
    def __load_mixed_dataset(file, output_index, sep=DEFAULT_DATASET_SEP):
        """Load a mixed data set.

        Inputs and output are in the same file.

        :param file: Name of the file containing the input and output data
        :type file: :py:class:`str`
        :param output_index: Index of the column containing the outuput data
        :type output_index: :py:class:`int`
        :param sep: Separator between columns, defaults to
            :py:attr:`~base.DEFAULT_DATASET_SEP`
        :type sep: :py:class:`str`, optional
        :raises TypeError: If *sep* is not a string
        :raises RuntimeError: If any input value is not numeric or there is any
            missing data
        :return: A tuple containing a :py:class:`~pandas.DataFrame` with the
            input columns and a :py:class:`~pandas.Series` with the output
            column
        :rtype: :py:class:`tuple`
        """
        # Check sep
        if not isinstance(sep, str):
            raise TypeError("sep must be a string")

        dataframe = read_csv(file, sep=sep, header=None)

        # Check if there is any missing data
        if Dataset.__has_missing_data(dataframe):
            raise RuntimeError(f"Missing data in {file}")

        inputs_df, output_s = Dataset.__split_input_output(
            dataframe, output_index)

        # Check if all inputs are numeric
        if not Dataset.__is_numeric(inputs_df):
            raise RuntimeError("Input data must contain only numerical data "
                               f"in {file}")

        return inputs_df, output_s

    @staticmethod
    def __load_split_dataset(*files, sep=DEFAULT_DATASET_SEP):
        """Load a separated data set.

        Inputs and output are in separated files. If the output file has more
        than one column, only the first column is read.

        :param files: Tuple of files containing the dataset. The first file
            contains the input columns and the second one the output column
        :type files: :py:class:`tuple` of :py:class:`str`
        :param sep: Separator between columns, defaults to
            :py:attr:`~base.DEFAULT_DATASET_SEP`
        :type sep: :py:class:`str`, optional
        :raises TypeError: If *sep* is not a string
        :raises RuntimeError: If any input value is not numeric, if there is
            any missing data, or if the *files* do not have the same number of
            rows
        :return: A tuple containing a :py:class:`~pandas.DataFrame` with the
            input columns and a :py:class:`~pandas.Series` with the output
            column
        :rtype: :py:class:`tuple`
        """
        # Check files
        if len(files) < 2:
            raise TypeError("A minimum of two files (inputs and outpus) "
                            "are needed (output_index is None)")

        # Check sep
        if not isinstance(sep, str):
            raise TypeError("sep must be a string")

        inputs_df = read_csv(files[0], sep=sep, header=None)

        # Check if there is any missing data
        if Dataset.__has_missing_data(inputs_df):
            raise RuntimeError(f"Missing data in {files[0]}")

        # Check if all inputs are numeric
        if not Dataset.__is_numeric(inputs_df):
            raise RuntimeError("Input data must contain only numerical data "
                               f"in {files[0]}")

        output_df = read_csv(files[1], sep=sep, header=None)
        output_s = output_df.iloc[:, 0]

        # Check if there is any missing data
        if Dataset.__has_missing_data(output_s):
            raise RuntimeError(f"Missing data in {files[1]}")

        # Check that both dataframes have the same number of rows
        if not len(inputs_df.index) == len(output_s.index):
            raise RuntimeError(f"{files[0]} and {files[1]} do not have the "
                               "same number of rows")

        return inputs_df, output_s


class Species(Base):
    """Base class for all the species in culebra.

    Each individual evolved within a wrapper must belong to a species which
    constraints the individual parameter values.
    """

    def check(self, ind):
        """Check if an individual meets the constraints imposed by the species.

        This method must be overriden by subclasses to return a correct
        value.

        :param ind: The individual
        :type ind: Any subclass of :py:class:`~base.Individual`
        :return: `True` if the individual belong to the species. `False`
            otherwise
        :rtype: :py:class:`tuple`
        """
        raise NotImplementedError("The check method has not been implemented "
                                  f"in the {self.__class__.__name__} class")


class Fitness(Base, DeapFitness):
    """Define the base class for the fitness evaluation of an individual."""

    weights = ()
    """Weights for each one of the objectives."""

    names = ()
    """Names of the objectives."""

    def __init__(self, **params):
        """Create the Fitness object.

        Fitness objects are compared lexicographically. The comparison applies
        a similarity threshold to assume that two fitness values are similar
        (if their difference is lower then the similarity threshold).

        :param thresholds: Thresholds to assume if two fitness values are
            equivalent. If only a single value is provided, the same threshold
            will be used for all the objectives. A different threshold can be
            provided for each objective with a
            :py:class:`~collections.abc.Sequence`. Defaults to
            :py:attr:`~base.DEFAULT_FITNESS_THRESHOLD`
        :type thresholds: :py:class:`float` or a
            :py:class:`~collections.abc.Sequence` of :py:class:`float` numbers,
            optional
        :raises TypeError: If *thresholds* is not a :py:class:`float` value or
            a :py:class:`~collections.abc.Sequence` of :py:class:`float`
            numbers
        :raises ValueError: If a negative threshold is provided
        """
        super().__init__()

        # Get the fitness thresholds
        self.thresholds = params.pop('thresholds', DEFAULT_FITNESS_THRESHOLD)

    def eval(self, ind, dataset):
        """Evaluate an individual.

        This method must be overriden by subclasses to return a correct
        value.

        :param ind: The individual
        :type ind: Any subclass of :py:class:`~base.Individual`
        :param dataset: A dataset
        :type dataset: :py:class:`~base.Dataset`
        :raises NotImplementedError: if has not been overriden
        :return: The fitness of *ind*
        :rtype: :py:class:`tuple`
        """
        raise NotImplementedError("The evaluation operator has not been "
                                  "implemented")

    @property
    def n_obj(self):
        """Get the number of objectives to be optimized.

        :type: :py:class:`int`
        """
        return len(self.names)

    @property
    def thresholds(self):
        """Get and set the similarity thresholds.

        Applied to assume if two fitness values are equivalent.

        :getter: Return the similarity threshold for each objective
        :setter: Set the similarity thresholds. If only a single value is
            provided, the same threshold will be used for all the objectives.
            A different threshold can be provided for each objective with a
            :py:class:`~collections.abc.Sequence`

        :type: :py:class:`tuple` of :py:class:`float` numbers
        :raises TypeError: If set with a value which is not a
            :py:class:`float` value or a :py:class:`~collections.abc.Sequence`
            of :py:class:`float` numbers
        :raises ValueError: If a negative threshold is provided
        """
        return self._thresholds

    @thresholds.setter
    def thresholds(self, value):
        """Set the similarity thresholds.

        Applied to assume if two fitness values are equivalent.

        :param value: If only a single value is provided, the same threshold
            will be used for all the objectives. A different threshold can be
            provided for each objective with a
            :py:class:`~collections.abc.Sequence`
        :type value: :py:class:`float` or a
            :py:class:`~collections.abc.Sequence` of :py:class:`float`
            numbers
        :raises TypeError: If *value* is not a :py:class:`float` value or a
            :py:class:`~collections.abc.Sequence` of :py:class:`float`
            numbers
        :raises ValueError: If a negative threshold is provided
        """
        def check_threshold(val):
            """Check if the given value is a valid threshold.

            :param val: A scalar threshold
            :type val: :py:class:`float`
            :raises TypeError: If *val* is not a real value
            :raises ValueError: If *val* is negative
            """
            # Check that it is a real number
            if not isinstance(val, numbers.Real):
                raise TypeError("The fitness threshold must be a real number "
                                "or a sequence of real numbers")
            # Check that it is not positive
            if val < 0:
                raise ValueError("The fitness threshold can not be negative")

        # If value is not a sequence ...
        if not isinstance(value, Sequence):
            # Check the value
            check_threshold(value)
            # Use it for all the objectives
            self._thresholds = (value,) * self.n_obj
        else:
            if len(value) != self.n_obj:
                raise ValueError("Incorrect number of thresholds. The number "
                                 f"of objectives is {self.n_obj}")

            # Check the values
            for val in value:
                check_threshold(val)

            # Set the thresholds
            self._thresholds = tuple(value)

    def dominates(self, other, which=slice(None)):
        """Check if this fitness dominates another one.

        :param other: The other fitness
        :type other: Any :py:class:`~base.Fitness` subclass
        :param which: Slice indicating on which objectives the domination is
            tested. The default value is `slice(None)`, representing every
            objectives
        :type which: :py:class:`slice`
        :return: `True` if each objective of *self* is not strictly worse than
            the corresponding objective of *other* and at least one objective
            is strictly better
        :rtype: :py:class:`bool`
        """
        not_equal = False
        for self_wval, other_wval, threshold in zip(
                self.wvalues[which], other.wvalues[which],
                self.thresholds[which]):
            if self_wval > other_wval and self_wval-other_wval > threshold:
                not_equal = True
            elif self_wval < other_wval and other_wval-self_wval > threshold:
                return False
        return not_equal

    def __hash__(self):
        """Return the hash number for this individual."""
        return hash(self.wvalues)

    def __gt__(self, other):
        """Greater than operator.

        :param other: The other fitness
        :type other: Any :py:class:`~base.Fitness` subclass
        """
        return not self.__le__(other)

    def __ge__(self, other):
        """Greater than or equal to operator.

        :param other: The other fitness
        :type other: Any :py:class:`~base.Fitness` subclass
        """
        return not self.__lt__(other)

    def __le__(self, other):
        """Less than or equal to operator.

        :param other: The other fitness
        :type other: Any :py:class:`~base.Fitness` subclass
        """
        for self_wval, other_wval, threshold in zip(
                self.wvalues, other.wvalues, self.thresholds):
            if self_wval < other_wval and other_wval-self_wval > threshold:
                return True
            if self_wval > other_wval and self_wval-other_wval > threshold:
                return False
        return True

    def __lt__(self, other):
        """Less than operator.

        :param other: The other fitness
        :type other: Any :py:class:`~base.Fitness` subclass
        """
        for self_wval, other_wval, threshold in zip(
                self.wvalues, other.wvalues, self.thresholds):
            if self_wval < other_wval and other_wval-self_wval > threshold:
                return True
            if self_wval > other_wval and self_wval-other_wval > threshold:
                return False
        return False

    def __eq__(self, other):
        """Equal to operator.

        :param other: The other fitness
        :type other: Any :py:class:`~base.Fitness` subclass
        """
        for self_wval, other_wval, threshold in zip(
                self.wvalues, other.wvalues, self.thresholds):
            if abs(self_wval-other_wval) > threshold:
                return False
        return True

    def __ne__(self, other):
        """Not equal to operator.

        :param other: The other fitness
        :type other: Any :py:class:`~base.Fitness` subclass
        """
        return not self.__eq__(other)


class Individual(Base):
    """Base class for all the individuals."""

    def __init__(self, species, fitness):
        """Create an individual.

        :param species: The species the individual will belong to
        :type species: :py:class:`~base.Species`
        :param fitness: The fitness object for the individual
        :type fitness: Instance of any subclass of
            :py:class:`~base.Fitness`
        :raises TypeError: If any parameter type is wrong
        """
        if not isinstance(species, Species):
            raise TypeError("Not valid species")
        if not isinstance(fitness, Fitness):
            raise TypeError("Not valid fitness")

        # Assing the species
        self._species = species

        # Assing an empty fitness
        self.fitness = copy.copy(fitness)

    @property
    def species(self):
        """Get the individuals species.

        :type: :py:class:`~base.Species`
        """
        return self._species

    def dominates(self, other, which=slice(None)):
        """Dominate operator.

        :param other: Other individual
        :type other: :py:class:`~base.Individual`
        :param which: Slice object indicating on which objectives the
            domination is tested. The default value is `slice(None)`,
            representing every objectives.
        :type which: :py:class:`slice`
        :return: `True` if each objective of *self* is not strictly worse than
            the corresponding objective of *other* and at least one objective
            is strictly better.
        :rtype: :py:class:`bool`
        """
        return self.fitness.dominates(other.fitness, which)

    def crossover(self, other):
        """Cross this individual with another one.

        This method must be overriden by subclasses to return a correct
        value.

        :param other: The other individual
        :type other: :py:class:`~base.Individual`
        :raises NotImplementedError: if has not been overriden
        :return: The two offspring
        :rtype: :py:class:`tuple`
        """
        raise NotImplementedError("The crossover operator has not been "
                                  "implemented in the "
                                  f"{self.__class__.__name__} class")

    def mutate(self, indpb):
        """Mutate the individual.

        This method must be overriden by subclasses to return a correct
        value.

        :param indpb: Independent probability for each feature to be mutated.
        :type indpb: :py:class:`float`
        :raises NotImplementedError: if has not been overriden
        :return: The mutant
        :rtype: :py:class:`tuple`
        """
        raise NotImplementedError("The mutation operator has not been "
                                  "implemented in the "
                                  f"{self.__class__.__name__} class")

    def __reduce__(self):
        """Reduce the individual.

        :return: The reduction
        :rtype: :py:class:`tuple`
        """
        return (self.__class__,
                (self._species, self.fitness),
                self.__dict__)

    def __copy__(self):
        """Shallow copy the individual."""
        cls = self.__class__
        result = cls(self.species, self.fitness)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        """Deepcopy the individual.

        :param memo: Individual attributes
        :type memo: :py:class:`dict`
        :return: A deep copy of the individual
        :rtype: :py:class:`~base.Individual`
        """
        cls = self.__class__
        result = cls(self.species, self.fitness)
        result.__dict__.update(copy.deepcopy(self.__dict__, memo))
        return result

    def __hash__(self):
        """Return the hash number for this individual."""
        return self.__str__().__hash__()

    def __eq__(self, other):
        """Equality test.

        :param other: Other individual
        :type other: :py:class:`~base.Individual`
        :return: `True` if the individual codes the same features. `False`
            otherwise
        :rtype: :py:class:`bool`
        """
        return self.__hash__() == other.__hash__()

    def __ne__(self, other):
        """Not equality test.

        :param other: Other individual
        :type other: :py:class:`~base.Individual`
        :return: `False` if the individual codes the same features. `True`
            otherwise
        :rtype: :py:class:`bool`
        """
        return not self.__eq__(other)

    def __lt__(self, other):
        """Less than operator.

        :param other: Other individual
        :type other: :py:class:`~base.Individual`
        :return: `True` if the individual's fitness is less than the other's
            fitness
        :rtype: :py:class:`bool`
        """
        return self.fitness < other.fitness

    def __gt__(self, other):
        """Greater than operator.

        :param other: Other individual
        :type other: :py:class:`~base.Individual`
        :return: `True` if the individual's fitness is greater than the other's
            fitness
        :rtype: :py:class:`bool`
        """
        return self.fitness > other.fitness

    def __le__(self, other):
        """Less than or equal to operator.

        :param other: Other individual
        :type other: :py:class:`~base.Individual`
        :return: `True` if the individual's fitness is less than or equal to
            the other's fitness
        :rtype: :py:class:`bool`
        """
        return self.fitness <= other.fitness

    def __ge__(self, other):
        """Greater than or equal to operator.

        :param other: Other individual
        :type other: :py:class:`~base.Individual`
        :return: `True` if the individual's fitness is greater than or equal to
            the other's fitness
        :rtype: :py:class:`bool`
        """
        return self.fitness >= other.fitness


class Wrapper(Base):
    """Base class for all the wrapper methods."""

    stats_names = DEFAULT_WRAPPER_STATS_NAMES
    """Statistics calculated each time."""

    objective_stats = DEFAULT_WRAPPER_OBJECTIVE_STATS
    """Statistics calculated for each objective."""

    def __init__(self, individual_cls, species, **params):
        """Initialize the wrapper method.

        :param individual_cls: Individual representation.
        :type individual_cls: Any subclass of
            :py:class:`~base.Individual`
        :param species: The species the individual will belong to
        :type species: :py:class:`~base.Species`
        :param checkpoint_freq: Frequency for checkpointing, defaults to
            :py:attr:`~base.DEFAULT_WRAPPER_CHECKPOINT_FREQ`
        :type checkpoint_freq: :py:class:`int`, optional
        :param checkpoint_file: File path for checkpointing, defaults to
            :py:attr:`~base.DEFAULT_WRAPPER_CHECKPOINT_FILE`
        :type checkpoint_file: :py:class:`str`, optional
        :param random_seed: Random seed for the random generator, defaults to
            `None`
        :type random_seed: :py:class:`int`, optional
        :param verbose: Whether or not to log the statistics, defaults to
            :py:data:`__debug__`
        :type verbose: :py:class:`bool`
        :raises TypeError: If any parameter has a wrong type
        """
        # Check the individual class
        if not (isinstance(individual_cls, type) and
                issubclass(individual_cls, Individual)):
            raise TypeError("Not valid individual class")
        self._individual_cls = individual_cls

        # Check the species
        if not isinstance(species, Species):
            raise TypeError("Not valid species")
        self._species = species

        # Get the checkpoint frequency
        self.checkpoint_freq = params.pop('checkpoint_freq',
                                          DEFAULT_WRAPPER_CHECKPOINT_FREQ)

        # Get the checkpoint file
        self.checkpoint_file = params.pop('checkpoint_file',
                                          DEFAULT_WRAPPER_CHECKPOINT_FILE)

        # Set the random seed for the number generator
        self.random_seed = params.pop('random_seed', None)

        # Set the verbosity of the algorithm
        self.verbose = params.pop('verbose', __debug__)

        # Initialize statistics object
        self._stats = Statistics(self._get_fitness_values)

        # Configure the stats
        for name, func in self.objective_stats.items():
            self._stats.register(name, func, axis=0)

        # Initialize the toolbox of DEAP
        self._toolbox = Toolbox()

    @staticmethod
    def _get_fitness_values(ind):
        """Return the fitness values of an individual.

        DEAP :py:class:`~deap.tools.Statistics` class needs a function to
        obtain the fitness values of an individual.

        :param ind: The individual
        :type ind: Any subclass of :py:class:`~base.Individual`
        :return: The fitness values of *ind*
        :rtype: :py:class:`tuple`
        """
        return ind.fitness.values

    @property
    def objective_stats_names(self):
        """Return the names of the objectives stats.

        :type: :py:class:`tuple` of :py:class:`str`
        """
        return tuple(self.objective_stats.keys())

    @property
    def checkpoint_freq(self):
        """Get and set the checkpoint frequency.

        :getter: Return the checkpoint frequency
        :setter: Set a value for the checkpoint frequency
        :raises TypeError: If set to a value which is not an integer
        :raises ValueError: If set to a value which is not a positive number
        :type: :py:class:`int`
        """
        return self._checkpoint_freq

    @checkpoint_freq.setter
    def checkpoint_freq(self, value):
        """Set a value for the checkpoint frequency.

        :param value: New value for the checkpoint frequency
        :type value: :py:class:`int`
        :raises TypeError: If *value* is not an integer
        :raises ValueError: If *value* is not a positive number
        """
        # Check the value
        if not isinstance(value, numbers.Integral):
            raise TypeError("The checkpoint frequency must be an integer "
                            "number")
        if value <= 0:
            raise ValueError("The checkpoint frequency must be a positive "
                             "number")

        self._checkpoint_freq = value

    @property
    def checkpoint_file(self):
        """Get and set the checkpoint file path.

        :getter: Return the checkpoint file path
        :setter: Set a new value for the checkpoint file path
        :type: :py:class:`str`
        :raises TypeError: If set to a value which is not a string
        """
        return self._checkpoint_file

    @checkpoint_file.setter
    def checkpoint_file(self, value):
        """Set a value for the checkpoint file path.

        :param value: New value for the checkpoint file path
        :type value: :py:class:`str`
        :raises TypeError: If *value* is not a string
        """
        # Check the value
        if not isinstance(value, str):
            raise TypeError("The checkpoint file path must be a string")

        self._checkpoint_file = value

    @property
    def random_seed(self):
        """Get and set the initial random seed used by this wrapper.

        :getter: Return the seed
        :setter: Set a new value for the random seed
        :type: :py:class:`int`
        """
        return self._random_seed

    @random_seed.setter
    def random_seed(self, value):
        """Set the random seed for this wrapper.

        :param value: Random seed for the random generator
        :type value: :py:class:`int`
        """
        self._random_seed = value
        random.seed(self._random_seed)
        np.random.seed(self._random_seed)

    @property
    def verbose(self):
        """Get and set the verbosity of this wrapper.

        :getter: Return the verbosity
        :setter: Set a new value for the verbosity
        :type: :py:class:`bool`
        :raises TypeError: If set to a value which is not boolean
        """
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        """Set the verbosity of this wrapper.

        :param value: `True` of `False`
        :type value: :py:class:`bool`
        :raises TypeError: If *value* is not boolean
        """
        if not isinstance(value, bool):
            raise TypeError("Verbose must be a boolean value")
        self._verbose = value

    def _search(self):
        """Apply the search algorithm.

        This method must be overriden by subclasses to return a correct
        value.

        :raises NotImplementedError: if has not been overriden
        :return: The best individuals found
        :rtype: :py:class:`~deap.tools.HallOfFame` of individuals
        :return: A logbook with the statistics of the evolution
        :rtype: :py:class:`~deap.tools.Logbook`
        :return: The runtime of the algorithm
        :rtype: :py:class:`float`
        """
        raise NotImplementedError("The search method has not been implemented "
                                  f"in the {self.__class__.__name__} class")

    def __check_dataset_and_fitness(self, dataset, fitness):
        """Check the parameters before training or testing.

        :param dataset: A dataset
        :type dataset: :py:class:`~base.Dataset`
        :param fitness: Fitness used while training or testing
        :type fitness: Any subclass of :py:class:`~base.Fitness`
        :raises TypeError: If any parameter has a wrong type
        :raises RuntimeError: If the number of features in the dataset
            does not match that of the species.
        """
        # Check the fitness class
        if not isinstance(fitness, Fitness):
            raise TypeError("Not valid fitness class")

        # Check the dataset
        if not isinstance(dataset, Dataset):
            raise TypeError("Not valid dataset")

        # Check if the number of features in the dataset matches that in the
        # species
        if dataset.num_feats != self._species.num_feats:
            raise RuntimeError("The number of features in the dataset "
                               f"({dataset.num_feats}) is different than that "
                               "initialized in the species "
                               f"({self._species.num_feats})")

    def train(self, dataset, fitness):
        """Perform the feature selection process.

        :param dataset: Training dataset
        :type dataset: :py:class:`~base.Dataset`
        :param fitness: Fitness used while training
        :type fitness: Any subclass of :py:class:`~base.Fitness`
        :raises TypeError: If any parameter has a wrong type
        :raises RuntimeError: If the number of features in the training data
            does not match that of the species.
        :return: The best individuals found
        :rtype: :py:class:`~deap.tools.HallOfFame` of individuals
        :return: A logbook with the statistics of the evolution
        :rtype: :py:class:`~deap.tools.Logbook`
        :return: The runtime of the algorithm
        :rtype: :py:class:`float`
        """
        # Check the parameters
        self.__check_dataset_and_fitness(dataset, fitness)

        # Register the function to initialize new individuals in the toolbox
        self._toolbox.register("individual", self._individual_cls,
                               self._species, fitness)

        # Register the evaluation function
        self._toolbox.register("evaluate", fitness.eval, dataset=dataset)

        # Search the best solutions
        return self._search()

    def test(self, hof, dataset, fitness):
        """Apply the test data to the solutions found by the wrapper method.

        Update the solutions in *hof* with their test fitness.

        :param hof: The best individuals found
        :type hof: :py:class:`~deap.tools.HallOfFame` of individuals
        :param dataset: Test dataset
        :type dataset: :py:class:`~base.Dataset`
        :param fitness: Fitness used to evaluate the final solutions
        :type fitness: Any subclass of :py:class:`~base.Fitness`
        :raises TypeError: If any parameter has a wrong type
        :raises RuntimeError: If the number of features in the test data does
            not match that of the species.
        """
        # Check the parameters
        self.__check_dataset_and_fitness(dataset, fitness)

        # For each solution found
        for ind in hof:
            ind.fitness.setValues(fitness.eval(ind, dataset))

    def __reduce__(self):
        """Reduce the wrapper.

        :return: The reduction
        :rtype: :py:class:`tuple`
        """
        return (self.__class__, (self._individual_cls, self._species),
                self.__dict__)

    def __copy__(self):
        """Shallow copy the wrapper."""
        cls = self.__class__
        result = cls(self._individual_cls, self._species)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        """Deepcopy the wrapper.

        :param memo: Individual attributes
        :type memo: :py:class:`dict`
        :return: A deep copy of the individual
        :rtype: :py:class:`~base.Individual`
        """
        cls = self.__class__
        result = cls(self._individual_cls, self._species)
        result.__dict__.update(copy.deepcopy(self.__dict__, memo))
        return result


class Metrics(Base):
    """Some metrics about the selected features obtained by a wrapper.

    Evaluate the set of solutions found by a :py:class:`~base.Wrapper` and
    calculate some metrics about the frequency of each selected feature.
    """

    @staticmethod
    def relevance(individuals):
        """Return the relevance of the features selected by a wrapper method.

        :param individuals: Best individuals returned by the wrapper method.
        :type individuals: Any iterable type.
        :return: The relevance of each feature appearing in the individuals.
        :rtype: :py:class:`~pandas.Series`
        """
        # species of the individuals
        species = individuals[0].species

        # all relevances are initialized to 0
        relevances = dict((feat, 0) for feat in np.arange(
            species.min_feat, species.max_feat + 1))
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
    def rank(individuals):
        """Return the rank of the features selected by a wrapper method.

        :param individuals: Best individuals returned by the wrapper method.
        :type individuals: Any iterable type.
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
            the_rank = (2*index + n_feats - 1) / 2
            index += n_feats
            for feat in feats:
                ranks[feat] = the_rank

        return Series(ranks).sort_index()
