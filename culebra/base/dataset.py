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

"""Provides the :py:class:`~base.dataset.Dataset` class."""

import numbers
import numpy as np
from pandas import read_csv, to_numeric
from sklearn.utils.validation import check_random_state
from sklearn.model_selection import train_test_split

__author__ = 'Jesús González'
__copyright__ = 'Copyright 2021, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.1.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'

DEFAULT_SEP = '\\s+'
"""Default column separator used within the files."""


class Dataset:
    """Dataset handler."""

    def __init__(self):
        """Create an empty dataset."""
        self._data_X = self._data_y = np.arange(0, dtype=float)

    @property
    def num_feats(self):
        """Number of features in the dataset.

        :type: :py:class:`int`
        """
        if self.size == 0:
            return 0
        else:
            return self._data_X.shape[1]

    @property
    def size(self):
        """Number of samples in the dataset.

        :type: :py:class:`int`
        """
        return self._data_X.shape[0]

    @property
    def X(self):
        """Input data of the dataset.

        :type: :py:class:`numpy.ndarray`
        """
        return self._data_X

    @property
    def y(self):
        """Output data of the dataset.

        :type: :py:class:`numpy.ndarray`
        """
        return self._data_y

    def normalize(self, test=None):
        """Normalize the dataset between 0 and 1.

        If a test dataset is provided, both are taken into account to calculate
        the minimum and maximum value and then both are normalized.

        :param test: Test dataset or `None`, defaults to `None`
        :type test: :py:class:`~base.dataset.Dataset`, optional
        """
        if test is None:
            self._data_X -= self._data_X.min(axis=0)
            self._data_X /= self._data_X.max(axis=0)
        else:
            min_X = np.minimum(self._data_X.min(axis=0),
                               test._data_X.min(axis=0))
            self._data_X -= min_X
            test._data_X -= min_X

            max_X = np.maximum(self._data_X.max(axis=0),
                               test._data_X.max(axis=0))
            self._data_X /= max_X
            test._data_X /= max_X

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
        :rtype: :py:class:`~base.dataset.Dataset`
        """
        # Check num_feats
        if not isinstance(num_feats, numbers.Integral):
            raise TypeError("The number of random features should be an "
                            "integer number")
        if num_feats <= 0:
            raise ValueError("The number of random features should be greater "
                             "than 0")

        # Check the random seed
        rs = check_random_state(random_seed)

        # Create an empty dataset
        new = self.__class__()

        # Append num_feats random features to the input data
        new._data_X = np.concatenate((self.data_X,
                                      rs.rand(self.size, num_feats)), axis=1)

        # Copy the output data
        new._data_y = np.copy(self.data_y)

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
        :rtype: :py:class:`tuple` of :py:class:`~base.dataset.Dataset`
        """
        # Check test_prop
        if not isinstance(test_prop, float):
            raise TypeError("The test proportion must be a float value")

        # Check the output_index value
        if not 0 < test_prop < 1:
            raise ValueError("The test proportion must be in (0, 1)")

        # Check the random seed
        rs = check_random_state(random_seed)

        (training_X,
         test_X,
         training_y,
         test_y) = train_test_split(self._data_X, self._data_y,
                                    test_size=test_prop, stratify=self._data_y,
                                    random_state=rs)
        training = self.__class__()
        training._data_X = training_X
        training._data_y = training_y
        test = self.__class__()
        test._data_X = test_X
        test._data_y = test_y

        return training, test

    @classmethod
    def load(cls, *files, output_index=None, sep=DEFAULT_SEP):
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
            :py:attr:`~base.dataset.DEFAULT_SEP`
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
        :rtype: :py:class:`~base.dataset.Dataset`
        """
        # If inputs and output data are in separate files
        if output_index is None:
            # Load the training data
            data_X, data_y = Dataset.__load_split_dataset(*files, sep=sep)
        # If inputs and output data are in the same file
        else:
            # Load the training data
            (data_X, data_y) = Dataset.__load_mixed_dataset(
                files[0], output_index=output_index, sep=sep)

        # Replace output labels by int identifiers (only if output is not
        # numeric)
        data_y = Dataset.__labels_to_numeric(data_y)

        # Create an empty dataset
        dataset = cls()

        # Convert data to numpy ndarrays
        dataset._data_X = data_X.to_numpy(dtype=float)
        dataset._data_y = data_y.to_numpy()

        # Return the dataset
        return dataset

    @classmethod
    def load_train_test(cls, *files, test_prop=None, output_index=None,
                        sep=DEFAULT_SEP, normalize=False, random_feats=None,
                        random_seed=None):
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
            :py:attr:`~base.dataset.DEFAULT_SEP`
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
        :return: A :py:class:`tuple` of :py:class:`~base.dataset.Dataset`
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
    def __is_numeric(df):
        """Check if all the elements in a dataframe are numeric.

        :param df: A dataframe
        :type df: :py:class:`~pandas.DataFrame`
        :return: `True` if all the elements are numeric
        :rtype: :py:class:`bool`
        """
        return df.apply(lambda s:
                        to_numeric(s, errors='coerce').notnull().all()
                        ).all()

    @staticmethod
    def __has_missing_data(df):
        """Check if the dataframe has missing values.

        :param df: A dataframe
        :type df: :py:class:`~pandas.DataFrame`
        :return: `True` if there is any missing value
        :rtype: :py:class:`bool`
        """
        return df.isna().values.any()

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
        else:
            labels = np.unique(values)
            rep = dict(zip(labels, range(len(labels))))
            return data_s.replace(rep)

    @staticmethod
    def __split_input_output(df, output_index):
        """Split a dataframe into input and output data.

        :param df: A dataframe containing input and output data
        :type df: :py:class:`~pandas.DataFrame`
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
        if not -len(df.columns) <= output_index < len(df.columns):
            raise IndexError("Output index out of range")

        output_s = df.iloc[:, output_index]
        inputs_df = df
        inputs_df.drop(inputs_df.columns[output_index], axis=1, inplace=True)

        return inputs_df, output_s

    @staticmethod
    def __load_mixed_dataset(file, output_index, sep=DEFAULT_SEP):
        """Load a mixed data set.

        Inputs and output are in the same file.

        :param file: Name of the file containing the input and output data
        :type file: :py:class:`str`
        :param output_index: Index of the column containing the outuput data
        :type output_index: :py:class:`int`
        :param sep: Separator between columns, defaults to
            :py:attr:`~base.dataset.DEFAULT_SEP`
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

        df = read_csv(file, sep=sep, header=None)

        # Check if there is any missing data
        if Dataset.__has_missing_data(df):
            raise RuntimeError(f"Missing data in {file}")

        inputs_df, output_s = Dataset.__split_input_output(df, output_index)

        # Check if all inputs are numeric
        if not Dataset.__is_numeric(inputs_df):
            raise RuntimeError("Input data must contain only numerical data "
                               f"in {file}")

        return inputs_df, output_s

    @staticmethod
    def __load_split_dataset(*files, sep=DEFAULT_SEP):
        """Load a separated data set.

        Inputs and output are in separated files. If the output file has more
        than one column, only the first column is read.

        :param files: Tuple of files containing the dataset. The first file
            contains the input columns and the second one the output column
        :type files: :py:class:`tuple` of :py:class:`str`
        :param sep: Separator between columns, defaults to
            :py:attr:`~base.dataset.DEFAULT_SEP`
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
        if not (len(inputs_df.index) == len(output_s.index)):
            raise RuntimeError(f"{files[0]} and {files[1]} do not have the "
                               "same number of rows")

        return inputs_df, output_s

    def __setstate__(self, state):
        """Set the state of the dataset.

        :param state: The state
        :type state: :py:class:`dict`
        """
        self.__dict__.update(state)

    def __reduce__(self):
        """Reduce the dataset.

        :return: The reduction
        :rtype: :py:class:`tuple`
        """
        return (self.__class__, (), self.__dict__)

    def __repr__(self):
        """Return the dataset representation."""
        cls = self.__class__
        cls_name = cls.__name__
        properties = (
                p for p in dir(cls)
                if isinstance(getattr(cls, p), property) and
                not p.startswith('_')
                      )

        repr = cls_name
        sep = "("
        for p in properties:
            repr += sep + p + "=" + getattr(self, p).__str__()
            sep = ", "

        repr += ")"
        return repr
