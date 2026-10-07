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

import warnings
from os import PathLike
from copy import deepcopy
from collections import Counter
from collections.abc import Sequence
from functools import partial
from numbers import Number
from io import BytesIO, TextIOBase
from urllib.request import urlopen
from urllib.parse import urlparse
from urllib.error import URLError

import numpy as np
from numpy.typing import ArrayLike
from pandas import DataFrame, read_csv, concat, notna
from pandas.errors import EmptyDataError
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from ucimlrepo import fetch_ucirepo
from scipy.io import loadmat
from imblearn.over_sampling import RandomOverSampler, SMOTE

from culebra.abc import Base
from culebra.checker import check_int, check_float, check_sequence
from .constants import (
    DEFAULT_SEP,
    DEFAULT_OUTLIER_PROPORTION,
    DEFAULT_SMOTE_NUM_NEIGHBORS
)

FilePath = str | PathLike[str]
Url = str


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2026, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.6.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


class Dataset(Base):
    """Dataset handler.

    Datasets can be loaded from local files or URLs. Their attributes are:

    * :attr:`~culebra.tools.Dataset.num_feats`: Number of input features
    * :attr:`~culebra.tools.Dataset.size`: Number of samples
    * :attr:`~culebra.tools.Dataset.inputs`: Input data
    * :attr:`~culebra.tools.Dataset.outputs`: Output data
    """

    def __init__(
        self,
        inputs: ArrayLike,
        outputs: ArrayLike
    ) -> None:
        """Create a dataset from input and output data.

        Both *inputs* and *outputs* must be array-like objects, such as NumPy
        arrays, lists of lists, tuples of tuples, pandas ``Series`` or pandas
        ``DataFrame`` objects.

        *inputs* must represent a two-dimensional structure where each row is
        a sample and each column is a feature.

        *outputs* may be either one-dimensional or two-dimensional. When a
        two-dimensional structure is provided, only its first column is used,
        assuming a single output value per sample.

        :param inputs: Input samples. Rows correspond to samples and columns
            correspond to features.
        :type inputs: numpy.typing.ArrayLike

        :param outputs: Output values associated with the input samples. It
            may be one-dimensional or two-dimensional. If it is
            two-dimensional, only the first column is considered.
        :type outputs: numpy.typing.ArrayLike

        :raises ValueError: If *inputs* and *outputs* do not contain the same
            number of samples.
        :raises ValueError: If no input samples are provided.
        :raises ValueError: If no output values are provided.
        """
        # Init the superclass
        super().__init__()

        try:
            inputs_df = DataFrame(inputs)
        except ValueError as e:
            raise ValueError("Invalid inputs") from e

        try:
            outputs_df = DataFrame(outputs)
        except ValueError as e:
            raise ValueError("Invalid outputs") from e

        if inputs_df.empty:
            raise ValueError("No inputs have been provided")
        if outputs_df.shape[1] == 0:
            raise ValueError("No outputs have been provided")

        outputs_first_column_df = outputs_df.iloc[:, [0]]
        if len(inputs_df) != len(outputs_first_column_df):
            raise ValueError(
                "The inputs and output do not have the same number of rows"
            )

        self._inputs = Dataset._categorical_to_numeric(
            inputs_df
        ).to_numpy(dtype=float)
        self._outputs = Dataset._categorical_to_numeric(
            outputs_first_column_df
        ).to_numpy().ravel()

    @property
    def num_feats(self) -> int:
        """Number of features in the dataset.

        :rtype: int
        """
        return 0 if self.size == 0 else self._inputs.shape[1]

    @property
    def size(self) -> int:
        """Number of samples in the dataset.

        :rtype: int
        """
        return self._inputs.shape[0]

    @property
    def inputs(self) -> np.ndarray:
        """Input data of the dataset.

        :rtype: ~numpy.ndarray
        """
        return self._inputs

    @property
    def outputs(self) -> np.ndarray:
        """Output data of the dataset.

        :rtype: ~numpy.ndarray
        """
        return self._outputs

    @classmethod
    def from_text(
        cls,
        *files: tuple[FilePath | Url | TextIOBase],
        output_index: int | None = None,
        sep: str = DEFAULT_SEP
    ) -> None:
        """Load a dataset from one or two text files.

        Datasets can be organized in only one file or in two files. If only one
        file is used, then *output_index* must be used to indicate which column
        stores the output values. If *output_index* is omitted, it will be
        assumed that the dataset is composed by two consecutive files, the
        first one containing the input columns and the second one storing the
        output column. Only the first column of the second file will be loaded
        in this case (just one output value per sample).

        :param files: Files containing the dataset. If *output_index* is
            omitted, two files are necessary, the first one containing
            the input columns and the second one containing the output column.
            Otherwise, only one file will be used to access to the whole
            dataset (input and output columns)
        :type files: tuple[str | ~os.PathLike[str] | ~io.TextIOBase]
        :param output_index: If the dataset is provided with only one file,
            this parameter indicates which column in the file does contain the
            output values. Otherwise this parameter must be omitted (set to
            :data:`None`) to express that inputs and ouputs are stored in
            two different files. Its default value is :data:`None`
        :type output_index: int
        :param sep: Column separator used within the files. Defaults to
            :attr:`~culebra.tools.DEFAULT_SEP`
        :type sep: str
        :raises ValueError: If *files* is empty
        :raises TypeError: If *output_index* is not :data:`None` or
            :class:`int`
        :raises TypeError: If *sep* is not a string
        :raises IndexError: If *output_index* is out of range
        :raises RuntimeError: If *output_index* is :data:`None` and only
            one file is provided
        :raises RuntimeError: When loading a dataset composed of two files, if
            the file containing the input columns and the file containing the
            output column do not have the same number of rows.
        :raises RuntimeError: If any file is empty
        :return: The dataset
        :rtype: ~culebra.tools.Dataset
        """
        # If no files are provided
        if len(files) == 0:
            raise ValueError("No files are provided")

        # If inputs and output data are in separate files
        if output_index is None:
            if len(files) < 2:
                raise RuntimeError(
                    "Only one file is provided and output_index is None"
                )

            # Load the dataset
            inputs_df = Dataset._text_to_dataframe(files[0], sep=sep)
            outputs_df = Dataset._text_to_dataframe(files[1], sep=sep)
        # If inputs and output data are in the same file
        else:
            # Load the dataset
            inputs_df, outputs_df = Dataset._separate_input_output(
                Dataset._text_to_dataframe(files[0], sep=sep),
                output_index
            )

        try:
            return cls(inputs_df, outputs_df)
        except ValueError as e:
            raise RuntimeError(str(e)) from e

    @classmethod
    def from_uci(
        cls,
        name: str | None = None,
        id_number: int | None = None
    ) -> Dataset:
        """Load a dataset from the UCI ML repository.

        The dataset can be identified by either its *id_number* or its *name*,
        but only one of these should be provided.

        If the dataset has more than one output column, only the first column
        is considered.

        :param name: Dataset name, or substring of name, optional
        :type name: str
        :param id_number: Dataset ID for UCI ML Repository, optional
        :type id_number: int

        :raises RuntimeError: If the dataset can not be loaded
        :return: The dataset
        :rtype: ~culebra.tools.Dataset
        """
        try:
            uci_dataset = fetch_ucirepo(name, id_number)
            return cls(
                uci_dataset.data.features,
                uci_dataset.data.targets
            )
        except Exception as e:
            raise RuntimeError(str(e)) from e

    @classmethod
    def from_mat(
        cls,
        src: FilePath | Url | BytesIO,
        inputs_key: str = 'X',
        outputs_key: str = 'Y'
    ) -> None:
        """Load a dataset from a mat file.

        :param src: Source
        :type src: str | ~os.PathLike[str] | io.BytesIO
        :param inputs_key: Key to access the dataset inputs. Defaults to 'X'
        :type inputs_key: str
        :param outputs_key: Key to access the dataset oututs. Defaults to 'Y'
        :type outputs_key: str
        :raises ValueError: If *src* is not a valid source
        :raises ValueError: If either *inputs_key* or *outputs_key* is not a
            valid key
        :return: The dataset
        :rtype: ~culebra.tools.Dataset
        """
        try:
            if (
                isinstance(src, str) and
                (parsed := urlparse(src)).scheme in ('http', 'https') and
                parsed.netloc
            ):
                with urlopen(src) as response:
                    mat = loadmat(BytesIO(response.read()))
            else:
                mat = loadmat(src)
        except (ValueError, URLError, FileNotFoundError) as e:
            raise ValueError(f"Invalid mat source '{src}'") from e

        # Check the key
        for key in (inputs_key, outputs_key):
            if key not in mat:
                raise ValueError(f"Bad key '{key}'")

        return cls(mat[inputs_key], mat[outputs_key])

    def to_text(
        self,
        filename: FilePath,
        sep: str = DEFAULT_SEP
    ) -> None:
        """Save the dataset to a text file.

        :param filename: Destination file name
        :type filename: ~os.PathLike[str]
        :param sep: Column separator used within the files. Defaults to
            :attr:`~culebra.tools.DEFAULT_SEP`
        :type sep: str
        """
        # Fallback if a regex-style whitespace separator is passed
        if sep == DEFAULT_SEP:
            sep = " "

        # Stack inputs and outputs horizontally (outputs becomes the last
        # column)
        # We cast to 'object' type so outputs preserve their native types
        combined_data = np.column_stack(
            (self.inputs, self.outputs.astype(object))
        )

        # 3. Save to a plain text file
        # fmt="%s" will now call the string representation of each native type,
        # preventing integers (like 1) from being written as floats (like 1.0)
        np.savetxt(filename, combined_data, delimiter=sep, fmt="%s")

    def normalize(self) -> Dataset:
        """Normalize the dataset between 0 and 1.

        :return: A normalized dataset
        :rtype: ~culebra.tools.Dataset
        """
        normalized_inputs = MinMaxScaler().fit(
            self._inputs
        ).transform(self._inputs)
        return Dataset(normalized_inputs, self.outputs)

    def scale(self) -> Dataset:
        """Scale features robust to outliers.

        :return: A scaled dataset
        :rtype: ~culebra.tools.Dataset
        """
        scaled_inputs = RobustScaler().fit(
            self._inputs
        ).transform(self._inputs)
        return Dataset(scaled_inputs, self.outputs)

    def drop_missing(self) -> Dataset:
        """Drop samples with missing values.

        :return: A clean dataset
        :rtype: ~culebra.tools.Dataset
        """
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

        clean_inputs = np.delete(
            self.inputs, samples_to_be_dropped, axis=0
        )

        clean_outputs = np.delete(
            self.outputs, samples_to_be_dropped, axis=0
        )

        return Dataset(clean_inputs, clean_outputs)

    def remove_outliers(
        self,
        prop: float = DEFAULT_OUTLIER_PROPORTION,
        random_seed: int | None = None
    ) -> Dataset:
        """Remove the outliers.

        :param prop: Expected outlier proportion por class, defaults to
            :attr:`~culebra.tools.DEFAULT_OUTLIER_PROPORTION`
        :type prop: float
        :param random_seed: Random seed for the random generator, defaults to
            :data:`None`
        :type random_seed: int
        :return: A clean dataset
        :rtype: ~culebra.tools.Dataset
        """
        # Check the random seed
        if random_seed is None:
            random_seed = np.random.default_rng().integers(0, 2**32 - 1)
        else:
            random_seed = check_int(random_seed, "random seed")

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
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    module=r"sklearn\.neighbors\..*",
                    message= (
                        r"n_neighbors \(\d+\) is greater than the total "
                        "number of samples \(\d+\).*"
                    )
                )

                for detector in detectors:
                    majority_voting += (detector.fit_predict(inputs_class) < 0)

            # Get the outliers indices
            outlier_indices = majority_voting > detection_th

            # Remove the outliers
            inputs_class = np.delete(inputs_class, outlier_indices, 0)

            filtered_inputs.append(inputs_class)
            filtered_outputs.append([class_label]*len(inputs_class))

        clean_inputs = np.vstack(filtered_inputs)
        clean_outputs = np.concatenate(filtered_outputs)
        return Dataset(clean_inputs, clean_outputs)

    def oversample(
        self,
        n_neighbors: int = DEFAULT_SMOTE_NUM_NEIGHBORS,
        random_seed: int | None = None
    ) -> Dataset:
        """Oversample all classes but the majority class.

        All classes but the majority class are oversampled to equal the number
        of samples of the majority class.
        :class:`~imblearn.over_sampling.SMOTE` is used for oversampling, but
        if any class has less than *n_neighbors* samples,
        :class:`~imblearn.over_sampling.RandomOverSampler` is first applied

        :param n_neighbors: Number of neighbors for
            :class:`~imblearn.over_sampling.SMOTE`, defaults to
            :attr:`~culebra.tools.DEFAULT_SMOTE_NUM_NEIGHBORS`
        :type n_neighbors: int
        :param random_seed: Random seed for the random generator, defaults to
            :data:`None`
        :type random_seed: int
        :return: An oversampled dataset
        :rtype: ~culebra.tools.Dataset
        """
        # Check the random seed
        if random_seed is None:
            random_seed = np.random.default_rng().integers(0, 2**32 - 1)
        else:
            random_seed = check_int(random_seed, "random seed")

        # Number of samples per class
        samples_per_class = Counter(self.outputs)

        # If any class has less than n_neighbors samples, RandomOverSampler
        # should be applied for all classes to have a minimum of n_neighbors
        # samples
        if any(count <= n_neighbors for count in samples_per_class.values()):
            # Define a sampling strategy to assure a minimum of
            # n_neighbos por class
            for key, val in samples_per_class.items():
                if val <= n_neighbors:
                    samples_per_class[key] = n_neighbors + 1

            # Create a RandomOverSampler instance
            random_over_sampler = RandomOverSampler(
                sampling_strategy=samples_per_class,
                random_state=random_seed
            )

            (
                resampled_inputs,
                resampled_outputs
            ) = random_over_sampler.fit_resample(self.inputs, self.outputs)
        else:
            # Keep the current dataset
            resampled_inputs, resampled_outputs = self.inputs, self.outputs

        # Apply SMOTE
        (
            resampled_inputs,
            resampled_outputs
        ) = SMOTE(
            k_neighbors=n_neighbors,
            random_state=random_seed
        ).fit_resample(resampled_inputs, resampled_outputs)

        return Dataset(resampled_inputs, resampled_outputs)

    def select_features(self, feats: Sequence[int]) -> Dataset:
        """Return a new dataset only with some selected features.

        :param feats: Indices of the selected features
        :type feats: ~collections.abc.Sequence[int]
        :return: The new dataset
        :rtype: ~culebra.tools.Dataset
        """
        feats = check_sequence(
            feats,
            "selected feature indices",
            item_checker=partial(check_int, ge=0, lt=self.num_feats)
        )

        return Dataset(self.inputs[:, feats], self.outputs)

    def append_random_features(
            self,
            num_feats: int,
            random_seed: int | None = None
    ) -> Dataset:
        """Return a new dataset with some random features appended.

        :param num_feats: Number of random features to be appended (greater
            than 0)
        :type num_feats: int
        :param random_seed: Random seed for the random generator, defaults to
            :data:`None`
        :type random_seed: int
        :raises TypeError: If the number of random features is not an integer
        :raises ValueError: If the number of random features not greater than
            0
        :return: The new dataset
        :rtype: ~culebra.tools.Dataset
        """
        # Check num_feats
        num_feats = check_int(num_feats, "number of features", gt=0)

        # Check the random seed
        if random_seed is None:
            random_generator = np.random.default_rng()
        else:
            random_generator = np.random.default_rng(random_seed)

        # Append num_feats random features to the input data
        new_inputs = np.concatenate(
            (self.inputs, random_generator.random((self.size, num_feats))),
            axis=1
        )

        # Return the new dataset
        return Dataset(new_inputs, self.outputs)

    def split(
            self,
            test_prop: float,
            random_seed: int | None = None
    ) -> tuple[Dataset, Dataset]:
        """Split the dataset.

        :param test_prop: Proportion of the dataset used as test data.
            The remaining samples will be returned as training data
        :type test_prop: float
        :param random_seed: Random seed for the random generator, defaults to
            :data:`None`
        :type random_seed: int
        :raises TypeError: If *test_prop* is not :data:`None` or
            :class:`float`
        :raises ValueError: If *test_prop* is not in (0, 1)
        :return: The training and test datasets
        :rtype: tuple[~culebra.tools.Dataset]
        """
        # Check test_prop
        test_prop = check_float(test_prop, "test proportion", gt=0, lt=1)

        # Check the random seed
        if random_seed is None:
            random_seed = np.random.default_rng().integers(0, 2**32 - 1)
        else:
            random_seed = check_int(random_seed, "random seed")

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
            random_state=random_seed
        )
        training = Dataset(training_inputs, training_outputs)
        test = Dataset(test_inputs, test_outputs)

        return training, test

    @staticmethod
    def _categorical_to_numeric(dataframe: DataFrame) -> DataFrame:
        """Replace categorical values by numeric values.

        :param dataframe: A dataframe
        :type dataframe: ~pandas.DataFrame
        :return: A dataframe with numerical values
        :rtype: ~pandas.DataFrame
        """
        columns_to_concat = []

        for col_name in dataframe:
            col = dataframe[col_name]

            # If any column value is not numeric
            if col.map(
                lambda x: notna(x) and not isinstance(x, Number)
            ).any():
                labels = col.dropna().unique()
                rep = {val: i for i, val in enumerate(labels)}
                # Keep the column name
                new_col = col.map(rep)
                columns_to_concat.append(new_col)
            else:
                # Is already numeric
                columns_to_concat.append(col)

        output_df = concat(columns_to_concat, axis=1)

        return output_df

    @staticmethod
    def _separate_input_output(
            data: DataFrame,
            output_index: int
    ) -> tuple[DataFrame, DataFrame]:
        """Separate a dataframe into input and output data.

        :param data: A dataframe containing input and output data
        :type data: ~pandas.DataFrame
        :param output_index: Index of the column containing the outuput data
        :type output_index: int
        :raises TypeError: If *output_index* is not an integer value
        :raises IndexError: If *output_index* is out of range
        :return: The inputs and outputs in separated DataFrames
        :rtype: tuple[~pandas.DataFrame, ~pandas.DataFrame]
        """
        # Check the type of output_index
        output_index = check_int(
            output_index,
            "output index",
            ge=-len(data.columns),
            lt=len(data.columns)
        )

        outputs_df = data.iloc[:, [output_index]]
        inputs_df = data.drop(data.columns[output_index], axis=1)

        return inputs_df, outputs_df

    @staticmethod
    def _text_to_dataframe(
        path: FilePath | Url | TextIOBase,
        sep: str = DEFAULT_SEP
    ) -> DataFrame:
        """Load a dataframe.

        :param path: Path to the file contining the data
        :type path: str | ~os.PathLike[str] | ~io.TextIOBase
        :param sep: Separator between columns
        :type sep: str
        :return: The dataframe
        :rtype: ~pandas.DataFrame
        """
        try:
            return read_csv(path, sep=sep, header=None)
        except TypeError as error:
            raise TypeError(f"Invalid separator: {sep}") from error
        except EmptyDataError as error:
            raise RuntimeError(f"No data in '{path}'") from error
        except FileNotFoundError as error:
            raise RuntimeError(f"Can't access to '{path}'") from error

    def __copy__(self) -> Dataset:
        """Shallow copy the dataset.

        :return: The copied object
        :rtype: ~culebra.tools.Dataset
        """
        cls = self.__class__
        result = cls(self.inputs, self.outputs)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo: dict) -> Dataset:
        """Deepcopy the dataset.

        :param memo: Dataset attributes
        :type memo: dict
        :return:  The copied dataset
        :rtype: ~culebra.tools.Dataset
        """
        cls = self.__class__
        result = cls(
            deepcopy(self.inputs, memo),
            deepcopy(self.outputs, memo)
        )
        result.__dict__.update(
            deepcopy(
                self.__dict__,
                memo |
                {id(self.inputs): result.inputs} |
                {id(self.outputs): result.outputs}
            )
        )
        return result

    def __reduce__(self) -> tuple:
        """Reduce the dataset.

        :return: The reduction
        :rtype: tuple
        """
        return (
            self.__class__,
            (self.inputs, self.outputs),
            self.__dict__)

    @classmethod
    def __fromstate__(cls, state: dict) -> Dataset:
        """Return a dataset from a state.

        :param state: The state
        :type state: dict
        :return: The dataset
        :rtype: ~culebra.tools.Dataset
        """
        obj = cls(state['_inputs'], state['_outputs'])
        obj.__setstate__(state)
        return deepcopy(obj)


# Exported symbols for this module
__all__ = [
    'Dataset'
]
