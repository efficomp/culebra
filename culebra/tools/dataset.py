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

"""Dataset handler."""

from __future__ import annotations

from collections import Counter
from os import PathLike
from copy import deepcopy
from typing import Optional, Tuple, Union, TextIO

import numpy as np
from pandas import Series, DataFrame, read_csv
from pandas.errors import EmptyDataError
from sklearn.utils.validation import check_random_state
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from ucimlrepo import fetch_ucirepo
from imblearn.over_sampling import RandomOverSampler, SMOTE

from culebra.abc import Base
from culebra.checker import check_str, check_int, check_float

FilePath = Union[str, "PathLike[str]"]
Url = str


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2025, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.3.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


DEFAULT_SEP = '\\s+'
"""Default column separator used within dataset files."""


class Dataset(Base):
    """Dataset handler.

    Datasets can be loaded from local files or URLs using either the
    constructor or the :py:meth:`~culebra.tools.Dataset.load_train_test`
    class method. Their attributes are:

      * :py:attr:`~culebra.tools.Dataset.num_feats`: Number of input features
      * :py:attr:`~culebra.tools.Dataset.size`: Number of samples
      * :py:attr:`~culebra.tools.Dataset.inputs`: Input data
      * :py:attr:`~culebra.tools.Dataset.outputs`: Output data
    """

    def __init__(
        self,
        *files: FilePath | Url | TextIO,
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
        :type files: Sequence of path-like objects, urls or file-like objects,
            optional
        :param output_index: If the dataset is provided with only one file,
            this parameter indicates which column in the file does contain the
            output values. Otherwise this parameter must be set to
            :py:data:`None` to express that inputs and ouputs are stored in
            two different files. Its default value is :py:data:`None`
        :type output_index: :py:class:`int`, optional
        :param sep: Column separator used within the files. Defaults to
            :py:attr:`~culebra.tools.DEFAULT_SEP`
        :type sep: :py:class:`str`, optional
        :raises TypeError: If *output_index* is not :py:data:`None` or
            :py:class:`int`
        :raises TypeError: If *sep* is not a string
        :raises IndexError: If *output_index* is out of range
        :raises RuntimeError: If *output_index* is :py:data:`None` and only
            one file is provided
        :raises RuntimeError: When loading a dataset composed of two files, if
            the file containing the input columns and the file containing the
            output column do not have the same number of rows.
        :raises RuntimeError: If any file is empty
        :return: The dataset
        :rtype: :py:class:`~culebra.tools.Dataset`
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
                if len(files) < 2:
                    raise RuntimeError(
                        "Only one file is provided and output_index is None"
                    )

                # Load the training data
                data_x, data_y = Dataset.__load_split_dataset(*files, sep=sep)
            # If inputs and output data are in the same file
            else:
                # Load the training data
                (data_x, data_y) = Dataset.__load_mixed_dataset(
                    files[0], output_index=output_index, sep=sep
                )

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

    @classmethod
    def load_from_uci(
        cls,
        name: Optional[str] = None,
        id: Optional[int] = None,
    ) -> Dataset:
        """Load the dataset from the UCI ML repository.

        The dataset can be identified by either its id or its name, but only
        one of these should be provided.

        If the dataset has more than one output column, only the first column
        is considered.

        :param name: Dataset name, or substring of name
        :type name: :py:class:`str`
        :param id: Dataset ID for UCI ML Repository
        :type id: :py:class:`int`
        :raises RuntimeError: If the dataset can not be loaded
        :return: The dataset
        :rtype: :py:class:`~culebra.tools.Dataset`
        """
        dataset = None

        try:
            # Fetch the dataset
            uci_dataset = fetch_ucirepo(name, id)

            inputs_df = Dataset.__categorical_to_numeric(
                uci_dataset.data.features
            )
            output_s = Dataset.__categorical_to_numeric(
                uci_dataset.data.targets
            ).iloc[:, 0]

            # Check that both dataframes have the same number of rows
            if not len(inputs_df.index) == len(output_s.index):
                raise RuntimeError(
                    "The inputs and output do not have the same number of rows"
                )

            # Convert data to numpy ndarrays and create the dataset
            dataset = Dataset()
            dataset._inputs = inputs_df.to_numpy(dtype=float)
            dataset._outputs = output_s.to_numpy()
        except Exception as e:
            raise RuntimeError(str(e)) from e

        return dataset

    def normalize(self) -> Dataset:
        """Normalize the dataset between 0 and 1.

        :return: A normalized dataset
        :rtype: :py:class:`~culebra.tools.Dataset`
        """
        normalized_dataset = Dataset()
        normalized_dataset._inputs = MinMaxScaler().fit(
            self._inputs
        ).transform(self._inputs)
        normalized_dataset._outputs = deepcopy(self.outputs)
        return normalized_dataset

    def scale(self) -> Dataset:
        """Scale features robust to outliers.

        :return: A scaled dataset
        :rtype: :py:class:`~culebra.tools.Dataset`
        """
        scaled_dataset = Dataset()
        scaled_dataset._inputs = RobustScaler().fit(
            self._inputs
        ).transform(self._inputs)
        scaled_dataset._outputs = deepcopy(self.outputs)
        return scaled_dataset

    def drop_missing(self) -> Dataset:
        """Drop samples with missing values.

        :return: A clean dataset
        :rtype: :py:class:`~culebra.tools.Dataset`
        """
        clean_dataset = Dataset()

        # Samples with missing inputs
        samples_to_be_dropped = [
            sample[0] for sample in np.argwhere(np.isnan(self.inputs))
        ]

        # Samples with missing outputs
        samples_to_be_dropped += [
            sample[0] for sample in np.argwhere(np.isnan(self.outputs))
        ]

        # remove duplicated indices
        samples_to_be_dropped = list(set(samples_to_be_dropped))

        clean_dataset._inputs = np.delete(
            self.inputs, samples_to_be_dropped, axis=0
        )

        clean_dataset._outputs = np.delete(
            self.outputs, samples_to_be_dropped, axis=0
        )

        return clean_dataset

    def remove_outliers(
        self,
        prop: float = 0.05,
        random_seed: Optional[int] = None
    ) -> Dataset:
        """Remove the outliers.

        :param prop: Expected outlier proportion por class, defaults to 0.05
        :type prop: :py:class:`float`
        :param random_seed: Random seed for the random generator, defaults to
            :py:data:`None`
        :type random_seed: :py:class:`int`, optional
        :return: A clean dataset
        :rtype: :py:class:`~culebra.tools.Dataset`
        """
        # The outlier detectors
        detectors = [
            IsolationForest(contamination=prop, random_state=random_seed),
            LocalOutlierFactor(contamination=prop),
            OneClassSVM(nu=prop)
        ]

        # Detection threshold
        detection_th = len(detectors) / 2

        # Filtered samples
        filtered_inputs = []
        filtered_outputs = []

        # Outlier detection by class
        for class_label in np.unique(self.outputs):
            # Filter samples by class
            inputs_class = self.inputs[self.outputs == class_label]

            # Use majority voting among several detectors
            majority_voting = np.zeros((len(inputs_class),))

            # Apply the detectors
            for detector in detectors:
                majority_voting += (detector.fit_predict(inputs_class) < 0)

            # Get the outliers indices
            outlier_indices = majority_voting > detection_th

            # Remove the outliers
            inputs_class = np.delete(inputs_class, outlier_indices, 0)

            filtered_inputs.append(inputs_class)
            filtered_outputs.append([class_label]*len(inputs_class))

        clean_dataset = Dataset()
        clean_dataset._inputs = np.vstack(filtered_inputs)
        clean_dataset._outputs = np.concatenate(filtered_outputs)
        return clean_dataset

    def oversample(
        self,
        n_neighbors: Optional[int] = 5,
        random_seed: Optional[int] = None
    ) -> Dataset:
        """Oversample all classes but the majority class.

        All classes but the majority class are oversampled to equal the number
        of samples of the majority class.
        :py:class:`~imblearn.over_sampling.SMOTE` is used for oversampling, but
        if any class has less than *n_neighbors* samples,
        :py:class:`~imblearn.over_sampling.RandomOverSampler` is first applied

        :param n_neighbors: Number of neighbors for
            :py:class:`~imblearn.over_sampling.SMOTE`, defaults to 5.
        :type n_neighbors: :py:class:`int`, optional
        :param random_seed: Random seed for the random generator, defaults to
            :py:data:`None`
        :type random_seed: :py:class:`int`, optional
        :return: An oversampled dataset
        :rtype: :py:class:`~culebra.tools.Dataset`
        """
        # Number of samples per class
        samples_per_class = Counter(self.outputs)

        # If any class has less than n_neighbors samples, RandomOverSampler
        # should be applied for all classes to have a minimum of n_neighbors
        # samples
        if any(count <= n_neighbors for count in samples_per_class.values()):
            # Define a sampling strategy to assure a minimum of
            # n_neighbos por class
            for key, val in samples_per_class.items():
                if samples_per_class[key] <= n_neighbors:
                    samples_per_class[key] = n_neighbors + 1

            # Create a RandomOverSampler instance
            random_over_sampler = RandomOverSampler(
                sampling_strategy=samples_per_class,
                random_state=random_seed
            )

            # Oversample the current dataset
            resampled_dataset = Dataset()
            (
                resampled_dataset._inputs,
                resampled_dataset._outputs
            ) = random_over_sampler.fit_resample(self.inputs, self.outputs)
        else:
            # Keep the current dataset
            resampled_dataset = self

        # Apply SMOTE
        (
            resampled_dataset._inputs,
            resampled_dataset._outputs
        ) = SMOTE(
            k_neighbors=n_neighbors,
            random_state=random_seed
        ).fit_resample(resampled_dataset.inputs, resampled_dataset.outputs)

        return resampled_dataset

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
        :rtype: :py:class:`~culebra.tools.Dataset`
        """
        # Check num_feats
        num_feats = check_int(num_feats, "number of features", gt=0)

        # Check the random seed
        random_state = check_random_state(random_seed)

        # Create an empty dataset
        new_dataset = self.__class__()

        # Append num_feats random features to the input data
        new_dataset._inputs = np.concatenate(
            (self._inputs, random_state.rand(self.size, num_feats)), axis=1
        )

        # Copy the output data
        new_dataset._outputs = np.deepcopy(self._outputs)

        # Return the new dataset
        return new_dataset

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
        :rtype: :py:class:`tuple` of :py:class:`~culebra.tools.Dataset`
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

    @staticmethod
    def __categorical_to_numeric(dataframe: DataFrame) -> DataFrame:
        """Replace categorical values by numeric values.

        :param dataframe: A dataframe
        :type dataframe: :py:class:`~pandas.DataFrame`
        :return: A dataframe with numerical values
        :rtype: :py:class:`~pandas.DataFrame`
        """
        output_df = DataFrame()
        for col_name in dataframe:
            col_values = dataframe[col_name].to_numpy()

            # If the column values are not numeric
            if not np.issubdtype(col_values.dtype, np.number):
                labels = np.unique(col_values)
                rep = dict(zip(labels, range(len(labels))))
                output_df[col_name] = dataframe[col_name].map(Series(rep))
            else:
                output_df[col_name] = dataframe[col_name]

        return output_df

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
    def __load_dataframe(
        path: FilePath | Url | TextIO,
        sep: str = DEFAULT_SEP
    ) -> DataFrame:
        """Load a dataframe.

        Also replace categorical data by numerical data.

        :param path: Path to the file contining the data
        :type path: Path-like object, urls or file-like object
        :param sep: Separator between columns
        :type sep: :py:class:`str`
        """
        # Check sep
        sep = check_str(sep, "separator")

        df = None
        try:
            # Read the data
            df = Dataset.__categorical_to_numeric(
                read_csv(path, sep=sep, header=None)
            )
        except EmptyDataError as error:
            raise RuntimeError(f"No data in {path}") from error

        return df

    @staticmethod
    def __load_mixed_dataset(
        file: FilePath | Url | TextIO,
        output_index: int,
        sep: str = DEFAULT_SEP
    ) -> Tuple[DataFrame, Series]:
        """Load a mixed data set.

        Inputs and output are in the same file.

        :param file: Name of the file containing the input and output data
        :type file: Path-like object, url or file-like object
        :param output_index: Index of the column containing the outuput data
        :type output_index: :py:class:`int`
        :param sep: Separator between columns, defaults to
            :py:attr:`~culebra.tools.DEFAULT_SEP`
        :type sep: :py:class:`str`, optional
        :raises TypeError: If *sep* is not a string
        :return: A tuple containing a :py:class:`~pandas.DataFrame` with the
            input columns and a :py:class:`~pandas.Series` with the output
            column
        :rtype: :py:class:`tuple`
        """
        # Load the data
        dataframe = Dataset.__load_dataframe(file, sep)

        # Separate inputs and outputs
        inputs_df, output_s = Dataset.__split_input_output(
            dataframe, output_index
        )

        return inputs_df, output_s

    @staticmethod
    def __load_split_dataset(
            *files: FilePath | Url | TextIO,
            sep: str = DEFAULT_SEP
    ) -> Tuple[DataFrame, Series]:
        """Load a separated data set.

        Inputs and output are in separated files. If the output file has more
        than one column, only the first column is read.

        :param files: Tuple of files containing the dataset. The first file
            contains the input columns and the second one the output column
        :type files: Sequence of path-like objects, urls or file-like objects
        :param sep: Separator between columns, defaults to
            :py:attr:`~culebra.tools.DEFAULT_SEP`
        :type sep: :py:class:`str`, optional
        :raises TypeError: If *sep* is not a string
        :raises RuntimeError: If the *files* do not have the same number of
            rows
        :return: A tuple containing a :py:class:`~pandas.DataFrame` with the
            input columns and a :py:class:`~pandas.Series` with the output
            column
        :rtype: :py:class:`tuple`
        """
        # Load the input data
        inputs_df = Dataset.__load_dataframe(files[0], sep)

        # Load the output data
        output_df = Dataset.__load_dataframe(files[1], sep)
        output_s = output_df.iloc[:, 0]

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
