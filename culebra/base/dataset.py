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

"""Dataset handler."""

from __future__ import annotations
from copy import copy
from typing import Optional, Tuple
from pandas import Series, DataFrame, read_csv, to_numeric
from pandas.errors import EmptyDataError
from sklearn.utils.validation import check_random_state
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import RobustScaler
import numpy as np
from . import Base, check_str, check_int, check_float


__author__ = "Jesús González"
__copyright__ = "Copyright 2021, EFFICOMP"
__license__ = "GNU GPL-3.0-or-later"
__version__ = "0.1.1"
__maintainer__ = "Jesús González"
__email__ = "jesusgonzalez@ugr.es"
__status__ = "Development"


DEFAULT_SEP = "\\s+"
"""Default column separator used within dataset files."""


class Dataset(Base):
    """Dataset handler.

    Datasets can be loaded from local files or URLs using either the
    constructor or the :py:meth:`~base.Dataset.load_train_test`
    class method. Their attributes are:

      * :py:attr:`~base.Dataset.num_feats`: Number of input features
      * :py:attr:`~base.Dataset.size`: Number of samples
      * :py:attr:`~base.Dataset.inputs`: Input data
      * :py:attr:`~base.Dataset.outputs`: Output data
    """

    def __init__(
        self,
        *files: str,
        output_index: Optional[int] = None,
        sep: str = DEFAULT_SEP
    ) -> None:
        """Create a dataset.

        Datasets can be organized in only one file or in two files. If one
        file per dataset is used, then *output_index* must be used to indicate
        which column stores the output values. If *output_index* is set to
        :py:data:`None` (its default value), it will be assumed that the
        dataset is composed by two consecutive files, the first one containing
        the input columns and the second one storing the output column. Only
        the first column of the second file will be loaded in this case (just
        one output value per sample).

        If no files are provided, an empty dataset is returned.

        :param files: Files containing the dataset. If *output_index* is
            :py:data:`None`, two files are necessary, the first one containing
            the input columns and the second one containing the output column.
            Otherwise, only one file will be used to access to the whole
            dataset (input and output columns)
        :type files: Sequence of :py:class:`str`, path object or
            file-like objects, optional
        :param output_index: If the dataset is provided with only one file,
            this parameter indicates which column in the file does contain the
            output values. Otherwise this parameter must be set to
            :py:data:`None` to express that inputs and ouputs are stored in
            two different files. Its default value is :py:data:`None`
        :type output_index: :py:class:`int`, optional
        :param sep: Column separator used within the files. Defaults to
            :py:attr:`~base.DEFAULT_SEP`
        :type sep: :py:class:`str`, optional
        :raises TypeError: If *output_index* is not :py:data:`None` or
            :py:class:`int`
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
        # Init the superclass
        super().__init__()

        # If no files are provided
        if len(files) == 0:
            # An empty dataset is returned
            self._inputs = self._outputs = np.arange(0, dtype=float)
        else:
            # If inputs and output data are in separate files
            if output_index is None:
                # Load the training data
                data_x, data_y = Dataset.__load_split_dataset(*files, sep=sep)
            # If inputs and output data are in the same file
            else:
                # Load the training data
                (data_x, data_y) = Dataset.__load_mixed_dataset(
                    files[0], output_index=output_index, sep=sep
                )

            # Replace output labels by int identifiers (only if output is not
            # numeric)
            data_y = Dataset.__labels_to_numeric(data_y)

            # Convert data to numpy ndarrays
            self._inputs = data_x.to_numpy(dtype=float)
            self._outputs = data_y.to_numpy()

    @property
    def num_feats(self) -> int:
        """Get the number of features in the dataset.

        :type: :py:class:`int`
        """
        return 0 if self.size == 0 else self._inputs.shape[1]

    @property
    def size(self) -> int:
        """Get the number of samples in the dataset.

        :type: :py:class:`int`
        """
        return self._inputs.shape[0]

    @property
    def inputs(self) -> np.ndarray:
        """Get the input data of the dataset.

        :type: :py:class:`numpy.ndarray`
        """
        return self._inputs

    @property
    def outputs(self) -> np.ndarray:
        """Get the output data of the dataset.

        :type: :py:class:`numpy.ndarray`
        """
        return self._outputs

    def normalize(self, test: Optional[Dataset] = None) -> None:
        """Normalize the dataset between 0 and 1.

        If a test dataset is provided, both are taken into account to calculate
        the minimum and maximum value and then both are normalized.

        :param test: Test dataset or :py:data:`None`, defaults to
            :py:data:`None`
        :type test: :py:class:`~base.Dataset`, optional
        """
        if test is None:
            self._inputs -= self._inputs.min(axis=0)
            self._inputs /= self._inputs.max(axis=0)
        else:
            min_inputs = np.minimum(
                self._inputs.min(axis=0), test._inputs.min(axis=0)
            )
            self._inputs -= min_inputs
            test._inputs -= min_inputs

            max_inputs = np.maximum(
                self._inputs.max(axis=0), test._inputs.max(axis=0)
            )
            self._inputs /= max_inputs
            test._inputs /= max_inputs

    def robust_scale(self, test: Optional[Dataset] = None) -> None:
        """Scale features robust to outliers.

        If a test dataset is provided, both are taken into account and then
        both are scaled.

        :param test: Test dataset or :py:data:`None`, defaults to
            :py:data:`None`
        :type test: :py:class:`~base.Dataset`, optional
        """
        if test is None:
            inputs = self._inputs
        else:
            inputs = np.concatenate((self._inputs, test._inputs))

        transformer = RobustScaler().fit(inputs)
        self._inputs = transformer.transform(self._inputs)
        if test is not None:
            test._inputs = transformer.transform(test._inputs)

    def remove_outliers(self, test: Optional[Dataset] = None) -> None:
        """Remove the outliers.

        If a test dataset is provided, both are taken into account to remove
        the outliers. Outliers will be also removed fomr the test dataset.

        :param test: Test dataset or :py:data:`None`, defaults to
            :py:data:`None`
        :type test: :py:class:`~base.Dataset`, optional
        """
        if test is None:
            inputs = self._inputs
        else:
            inputs = np.concatenate((self._inputs, test._inputs))

        # The outlier detectors
        detectors = [
            IsolationForest(),
            LocalOutlierFactor(),
            OneClassSVM(nu=0.01)
        ]

        # Use majority voting among several detectors
        majority_voting = {}
        for i in range(len(inputs)):
            majority_voting[i] = 0

        # Apply the detectors
        for detector in detectors:
            for i in np.where(detector.fit_predict(inputs) < 0)[0]:
                majority_voting[i] += 1

        # Get the outliers
        threshold = len(detectors) / 2
        outlier_indices = [
            key for key in majority_voting.keys()
            if majority_voting[key] > threshold
        ]

        # Remove the outliers
        if test is None:
            self._inputs = np.delete(self._inputs, outlier_indices, 0)
            self._outputs = np.delete(self._outputs, outlier_indices, 0)
        else:
            train_outlier_indices = [
                index for index in outlier_indices if index < self.size
            ]
            test_outlier_indices = [
                index - self.size
                for index in outlier_indices if index >= self.size
            ]
            self._inputs = np.delete(self._inputs, train_outlier_indices, 0)
            self._outputs = np.delete(self._outputs, train_outlier_indices, 0)
            test._inputs = np.delete(test._inputs, test_outlier_indices, 0)
            test._outputs = np.delete(test._outputs, test_outlier_indices, 0)

    def append_random_features(
            self,
            num_feats: int,
            random_seed: Optional[int] = None
    ) -> Dataset:
        """Return a new dataset with some random features appended.

        :param num_feats: Number of random features to be appended
        :type num_feats: An :py:class:`int` greater than 0
        :param random_seed: Random seed for the random generator, defaults to
            :py:data:`None`
        :type random_seed: :py:class:`int`, optional
        :raises TypeError: If the number of random features is not an integer
        :raises ValueError: If the number of random features not greater than
            0
        :return: The new dataset
        :rtype: :py:class:`~base.Dataset`
        """
        # Check num_feats
        num_feats = check_int(num_feats, "number of features", gt=0)

        # Check the random seed
        random_state = check_random_state(random_seed)

        # Create an empty dataset
        new = self.__class__()

        # Append num_feats random features to the input data
        new._inputs = np.concatenate(
            (self._inputs, random_state.rand(self.size, num_feats)), axis=1
        )

        # Copy the output data
        new._outputs = np.copy(self._outputs)

        # Return the new dataset
        return new

    def split(
            self,
            test_prop: float,
            random_seed: Optional[int] = None
    ) -> Tuple[Dataset, Dataset]:
        """Split the dataset.

        :param test_prop: Proportion of the dataset used as test data.
            The remaining samples will be returned as training data
        :type test_prop: :py:class:`float`
        :param random_seed: Random seed for the random generator, defaults to
            :py:data:`None`
        :type random_seed: :py:class:`int`, optional
        :raises TypeError: If *test_prop* is not :py:data:`None` or
            :py:class:`float`
        :raises ValueError: If *test_prop* is not in (0, 1)
        :return: The training and test datasets
        :rtype: :py:class:`tuple` of :py:class:`~base.Dataset`
        """
        # Check test_prop
        test_prop = check_float(test_prop, "test proportion", gt=0, lt=1)

        # Check the random seed
        random_seed = check_random_state(random_seed)

        (
            training_inputs,
            test_inputs,
            training_outputs,
            test_outputs,
        ) = train_test_split(
            self._inputs,
            self._outputs,
            test_size=test_prop,
            stratify=self._outputs,
            random_state=random_seed,
        )
        training = self.__class__()
        training._inputs = training_inputs
        training._outputs = training_outputs
        test = self.__class__()
        test._inputs = test_inputs
        test._outputs = test_outputs

        return training, test

    @classmethod
    def load_train_test(
            cls,
            *files: str,
            test_prop: Optional[float] = None,
            output_index: Optional[int] = None,
            sep: str = DEFAULT_SEP,
            normalize: bool = False,
            random_feats: Optional[int] = None,
            random_seed: Optional[int] = None,
    ) -> Tuple[Dataset, Dataset]:
        """Load the training and test data.

        Datasets can be organized in only one file or in two files. If one
        file per dataset is used, then *output_index* must be used to indicate
        which column stores the output values. If *output_index* is set to
        :py:data:`None` (its default value), it will be assumed that each
        dataset is composed by two consecutive files, the first one containing
        the input columns and the second one storing the output column. Only
        the first column of the second file will be loaded (just one output
        value per sample).

        If *test_prop* is set a valid proportion, the training and test
        datasets will be generated splitting the first dataset provided.
        Otherwise a test dataset can be provided appending more files, but if
        it isn't, the training dataset is assumed to be also the test dataset.

        :param files: Files containing the data. If *output_index* is
            :py:data:`None`,
            two consecutive files are required for each dataset, otherwise,
            each file contains a whole dataset (input and output columns).
        :type files: Sequence of :py:class:`str`, path object or
            file-like objects
        :param test_prop: Proportion of the first dataset used as test data.
            The remaining samples will be used as training data.
        :type test_prop: :py:class:`float`, optional
        :param output_index: If datasets are provided with only one file,
            this parameter indicates which column in the file does contain the
            output values. Otherwise this parameter must be set to
            :py:data:`None` to express that inputs and ouputs are stored in
            two different files. Its default value is :py:data:`None`
        :type output_index: :py:class:`int`, optional
        :param sep: Column separator used within the files. Defaults to
            :py:attr:`~base.DEFAULT_SEP`
        :type sep: :py:class:`str`, optional
        :param normalize: If :py:data:`True`, datasets will be normalized
            into [0, 1]. Defaults to :py:data:`False`
        :type normalize: :py:class:`bool`
        :param random_feats: Number of random features to be appended, defaults
            to :py:data:`None`
        :type random_feats: An :py:class:`int` greater than 0, optional
        :param random_seed: Random seed for the random generator, defaults to
            :py:data:`None`
        :type random_seed: :py:class:`int`, optional
        :raises TypeError: If at least one dataset is not provided
        :raises TypeError: If *output_index* is not :py:data:`None` or
            :py:class:`int`
        :raises TypeError: If *test_prop* is not :py:data:`None` or
            :py:class:`float`
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
        training = cls(*files, output_index=output_index, sep=sep)

        # No test dataset by the moment...
        test = None

        # If the dataset must be split...
        if test_prop is not None:
            # Split the dataset
            training, test = training.split(test_prop, random_seed)

        # Else, if there are test data files...
        elif len(files) >= 2 * files_per_dataset:
            # Load the test dataset
            test = cls(
                *files[files_per_dataset:],
                output_index=output_index,
                sep=sep
            )

            # Check if training and test data have the same number of columns
            if training.num_feats != test.num_feats:
                raise RuntimeError(
                    "Training and test data do not have the same number of "
                    "features"
                )
        if normalize:
            training.normalize(test)

        if random_feats is not None:
            training = training.append_random_features(
                random_feats, random_seed)
            if test is not None:
                test = test.append_random_features(random_feats, random_seed)

        # The test dataset will be the same dataset by default
        if test is None:
            test = copy(training)

        return training, test

    @staticmethod
    def __is_numeric(dataframe: DataFrame) -> bool:
        """Check if all the elements in a dataframe are numeric.

        :param dataframe: A dataframe
        :type dataframe: :py:class:`~pandas.DataFrame`
        :return: :py:data:`True` if all the elements are numeric
        :rtype: :py:class:`bool`
        """
        return dataframe.apply(
            lambda s: to_numeric(s, errors="coerce").notnull().all()
        ).all()

    @staticmethod
    def __has_missing_data(dataframe: DataFrame) -> bool:
        """Check if a dataframe has missing values.

        :param dataframe: A dataframe
        :type dataframe: :py:class:`~pandas.DataFrame`
        :return: :py:data:`True` if there is any missing value
        :rtype: :py:class:`bool`
        """
        return dataframe.isna().values.any()

    @staticmethod
    def __labels_to_numeric(data_s: Series) -> Series:
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
            numeric_values = data_s
        else:
            labels = np.unique(values)
            rep = dict(zip(labels, range(len(labels))))
            numeric_values = data_s.replace(rep)

        return numeric_values

    @staticmethod
    def __split_input_output(
            data: DataFrame,
            output_index: int
    ) -> Tuple[DataFrame, Series]:
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
        output_index = check_int(
            output_index,
            "output index",
            ge=-len(data.columns),
            lt=len(data.columns)
        )

        output_s = data.iloc[:, output_index]
        inputs_df = data
        inputs_df.drop(inputs_df.columns[output_index], axis=1, inplace=True)

        return inputs_df, output_s

    @staticmethod
    def __load_mixed_dataset(
            file: str,
            output_index: int,
            sep: str = DEFAULT_SEP) -> Tuple[DataFrame, Series]:
        """Load a mixed data set.

        Inputs and output are in the same file.

        :param file: Name of the file containing the input and output data
        :type file: :py:class:`str`
        :param output_index: Index of the column containing the outuput data
        :type output_index: :py:class:`int`
        :param sep: Separator between columns, defaults to
            :py:attr:`~base.DEFAULT_SEP`
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
        sep = check_str(sep, "separator")

        try:
            dataframe = read_csv(file, sep=sep, header=None)
        except EmptyDataError as error:
            raise RuntimeError(f"No data in {file}") from error

        # Check if there is any missing data
        if Dataset.__has_missing_data(dataframe):
            raise RuntimeError(f"Missing data in {file}")

        inputs_df, output_s = Dataset.__split_input_output(
            dataframe, output_index)

        # Check if all inputs are numeric
        if not Dataset.__is_numeric(inputs_df):
            raise RuntimeError(
                f"Input data must contain only numerical data in {file}"
            )

        return inputs_df, output_s

    @staticmethod
    def __load_split_dataset(
            *files: str,
            sep: str = DEFAULT_SEP
    ) -> Tuple[DataFrame, Series]:
        """Load a separated data set.

        Inputs and output are in separated files. If the output file has more
        than one column, only the first column is read.

        :param files: Tuple of files containing the dataset. The first file
            contains the input columns and the second one the output column
        :type files: :py:class:`tuple` of :py:class:`str`
        :param sep: Separator between columns, defaults to
            :py:attr:`~base.DEFAULT_SEP`
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
            raise TypeError(
                "A minimum of two files (inputs and outpus) are needed "
                "(output_index is None)"
            )

        # Check sep
        sep = check_str(sep, "separator")

        try:
            inputs_df = read_csv(files[0], sep=sep, header=None)
        except EmptyDataError as error:
            raise RuntimeError(f"No data in {files[0]}") from error

        # Check if there is any missing data
        if Dataset.__has_missing_data(inputs_df):
            raise RuntimeError(f"Missing data in {files[0]}")

        # Check if all inputs are numeric
        if not Dataset.__is_numeric(inputs_df):
            raise RuntimeError(
                f"Input data must contain only numerical data in {files[0]}"
            )

        try:
            output_df = read_csv(files[1], sep=sep, header=None)
        except EmptyDataError as error:
            raise RuntimeError(f"No data in {files[1]}") from error

        output_s = output_df.iloc[:, 0]

        # Check if there is any missing data
        if Dataset.__has_missing_data(output_s):
            raise RuntimeError(f"Missing data in {files[1]}")

        # Check that both dataframes have the same number of rows
        if not len(inputs_df.index) == len(output_s.index):
            raise RuntimeError(
                f"{files[0]} and {files[1]} do not have the same number of "
                "rows"
            )

        return inputs_df, output_s


# Exported symbols for this module
__all__ = [
    'Dataset',
    'DEFAULT_SEP'
]
