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

"""Provides the :py:class:`~tools.config_manager.ConfigManager` class."""

import toml
from os import path
from pydoc import locate
from culebra.base.species import Species
from culebra.base.dataset import Dataset
from culebra.tools.experiment import Experiment
from culebra.tools.batch import Batch

__author__ = 'Jesús González'
__copyright__ = 'Copyright 2021, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.1.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


class ConfigSections:
    """Sections in the config file."""

    dataset = "dataset"
    """Section containing the dataset configuration."""

    wrapper = "wrapper"
    """Section containing the wrapper configuration."""

    batch = "batch"
    """Section containing the batch configuration."""

    species = "species"
    """Section containing the species configuration."""

    fitness = "fitness"
    """Section containing the fitness configuration."""

    training = "training"
    """Section containing the training configuration."""

    test = "test"
    """Section containing the test configuration."""

    classifier = "classifier"
    """Section containing the details of the classifier."""

    parameters = "parameters"
    """Section containing the parameters of any object or class."""


class ConfigKeys:
    """Keys in the config file."""

    test_prop = "test_prop"
    """Proportion of data used to test."""

    file = "file"
    """Inputs file name."""

    output_file = "output_file"
    """Output file name."""

    wrapper_cls = "wrapper_cls"
    """Key containing the wrapper class."""

    individual_cls = "individual_cls"
    """Key containing the individual class."""

    fitness_cls = "fitness_cls"
    """Key containing the fitness class."""

    classifier_cls = "classifier_cls"
    """Key containing the classifier class."""


class ConfigManager:
    """Manage the configuration file."""

    _toml_sep = "."
    """Section separator in TOML files."""

    _object_suffixes = ("_func", "_cls")
    """Key suffixes of values containing object names that should be replaced
    by the the object they name."""

    def __init__(self, config):
        """Create the manager from a TOML configuration file.

        :param config: Configuration file
        :type config: A path to a file
        """
        self._config_path = path.abspath(config)
        self._config = self.__replace_objects(toml.load(config))

    def load_dataset(self):
        """Parse the config parameters to load the training and test data.

        Once the configuration parameters are processed, this functions calls
        to :py:meth:`~base.dataset.Dataset.load_train_test` to load the
        data.

        :raises ValueError: If the structure of the config file is not valid.
        :return: A :py:class:`tuple` of :py:class:`~base.dataset.Dataset`
            containing the training and test datasets
        :rtype: :py:class:`tuple`
        """
        # Dataset parameters section name
        dataset_params_section = (
            ConfigSections.dataset + self._toml_sep +
            ConfigSections.parameters)

        # Get the dataset parameters
        dataset_params = self.__get_value(
            dataset_params_section, self._config, is_section=True)

        # Training dataset section name
        tr_dataset_section = (
            ConfigSections.dataset + self._toml_sep +
            ConfigSections.training)

        # Get the training dataset configuration
        tr_dataset_config = self.__get_value(
            tr_dataset_section, self._config, is_section=True, mandatory=True)

        # Get the training file names
        files = self.__get_dataset_files(
            tr_dataset_config, tr_dataset_section)

        # Check if the test data section should be considered
        # Get the dataset parameters
        test_prop_param = (
            dataset_params_section + self._toml_sep +
            ConfigKeys.test_prop)
        test_prop = self.__get_value(test_prop_param, self._config)

        # If test_prop is omitted, the test data must be loaded
        if test_prop is None:
            # Test dataset section name
            tst_dataset_section = (
                ConfigSections.dataset + self._toml_sep +
                ConfigSections.test)

            # Get the test dataset configuration
            tst_dataset_config = self.__get_value(
                tst_dataset_section, self._config, is_section=True)
            # Get the test file names
            files += self.__get_dataset_files(
                tst_dataset_config, tst_dataset_section)
        # Load and return the dataset
        return Dataset.load_train_test(*files, **dataset_params)

    def build_wrapper(self):
        """Return a wrapper method built from the configuration parameters.

        :raises ValueError: If the structure of the config file is not valid.
        :return: The wrapper object
        :rtype: A subclass of :py:class:`~base.wrapper.Wrapper`
        """
        # Get the wrapper configuration
        wrapper_config = self.__get_value(
            ConfigSections.wrapper, self._config, is_section=True,
            mandatory=True)
        # Get the wrapper class
        wrapper_cls = self.__get_value(
            ConfigKeys.wrapper_cls, wrapper_config,
            section_name=ConfigSections.wrapper, mandatory=True)

        # Get the individual class name
        individual_cls = self.__get_value(
            ConfigKeys.individual_cls, wrapper_config,
            section_name=ConfigSections.wrapper, mandatory=True)

        # Get the species parameters
        species_params = self.__get_value(
            ConfigSections.species, wrapper_config,
            section_name=ConfigSections.wrapper, is_section=True,
            mandatory=True)

        # Build the species
        species = Species(**species_params)

        # Get the wrapper parameters
        wrapper_params = self.__get_value(
            ConfigSections.parameters, wrapper_config,
            section_name=ConfigSections.wrapper, is_section=True)

        # Build the wrapper object
        return wrapper_cls(individual_cls, species, **wrapper_params)

    def build_fitness(self):
        """Return the training and test fitness objects.

        If the test fitness is omitted in the config file, the training
        fitness will also be returned as test fitness.

        :raises ValueError: If any mandatory value is missing
        :return: The training and test fitness
        :rtype: A :py:class:`tuple` composed by two instances of any subclass
            of :py:class:`~base.fitness.Fitness`
        """
        # Training fitness section name
        tr_fitness_section = (
            ConfigSections.fitness + self._toml_sep +
            ConfigSections.training)

        # Get the training fitness configuration
        tr_fitness_config = self.__get_value(
            tr_fitness_section, self._config, is_section=True, mandatory=True)

        # Get the training fitness
        tr_fitness = self.__build_fitness(
            tr_fitness_config, tr_fitness_section)

        # Test fitness section name
        tst_fitness_section = (
            ConfigSections.fitness + self._toml_sep + ConfigSections.test)

        # Get the test fitness
        tst_fitness_config = self.__get_value(
            tst_fitness_section, self._config, is_section=True)

        # If test fitness is omitted, the training fitness will be also used
        # to test
        if tst_fitness_config is None:
            tst_fitness = tr_fitness
        else:
            # Otherwise, get the test fitness
            tst_fitness = self.__build_fitness(
                tst_fitness_config, tst_fitness_section)

        return tr_fitness, tst_fitness

    def build_experiment(self):
        """Return an experiment to run the wrapper method.

        :raises ValueError: If any mandatory value is missing
        :return: The experiment
        :rtype: An :py:class:`~tools.experiment.Experiment` object
        """
        # Load the training and test data
        tr_data, tst_data = self.load_dataset()

        # Build the training and test fitness
        tr_fitness, tst_fitness = self.build_fitness()

        # Create the wrapper
        wrapper = self.build_wrapper()

        return Experiment(wrapper, tr_data, tr_fitness, tst_data, tst_fitness)

    def build_batch(self):
        """Return a batch of experiments to test the wrapper method.

        :raises ValueError: If any mandatory value is missing
        :return: The batch
        :rtype: A :py:class:`~tools.batch.Batch` object
        """
        # Get the wrapper configuration
        batch_config = self.__get_value(
            ConfigSections.batch, self._config, is_section=True)

        return Batch(self._config_path, **batch_config)

    def __replace_objects(self, config, key=None):
        """Replace all object names by the objects they name.

        Check all the keys in the config parameters and replaces the
        values of all the keys having a suffix in
        :py:attr:`~tools.config_manager.ConfigManager.object_suffixes`, which
        contain object names, by the objects they name.

        :param config: Configuration parameters
        :type config: Any valid TOML value
        :param key: A key, defaults to `None`
        :type key: :py:class:`str`, optional
        :raises ValueError: If any object can not be found.
        :return: The modiffied parameters
        :rtype: The same of *config*
        """
        if config is not None:
            if isinstance(config, dict):
                for key, value in config.items():
                    config[key] = self.__replace_objects(value, key)
            elif key is not None:
                for suffix in self._object_suffixes:
                    if key.endswith(suffix):
                        value = config
                        config = locate(value)
                        if config is None:
                            raise ValueError(f"Not valid value for {key}: "
                                             f"{value}")
                        break

        return config

    def __get_value(self, name, section_config, *, section_name=None,
                    is_section=False, mandatory=False):
        """Return a value from a section in a config file.

        :param name: Full name of the value.
        :type name: :py:class:`str`
        :param section_config: Section containing the value.
        :type section_config: :py:class:`dict`
        :param section_name: Name of the section containing the value, defaults
            to `None`
        :type section_name: :py:class:`str` or `None`
        :param is_section: `True` if the value is a section, defaults to
            `False`
        :type is_section: :py:class:`bool`
        :param mandatory: `True` if the value is mandatory, defaults to `False`
        :type mandatory: :py:class:`bool`
        :raises ValueError: If a mandatory value is missing
        :return: The value it it exists, or `None` otherwise
        :rtype: Any valid value in a TOML file
        """
        hierarchy = name.split(self._toml_sep)

        # The whole config file is used by default
        if section_config is None:
            value = self._config
        else:
            value = section_config
        for section in hierarchy:
            # Check the section exists
            if section in value.keys():
                value = value[section]
            else:
                if mandatory:
                    err_msg = "No "
                    if is_section:
                        err_msg += f"[{name}] section "
                    else:
                        err_msg += f"{name} key "

                    err_msg += "in "

                    if section_name is None:
                        err_msg += "the configuration file"
                    else:
                        err_msg += f"section [{section_name}]"

                    raise ValueError(err_msg)
                else:
                    return {} if is_section else None

        return value

    def __get_dataset_files(self, dataset_config, section_name):
        """Return the dataset file names.

        :param dataset_config: Section where the file names are defined
        :type dataset_config: :py:class:`dict`
        :param section_name: Name of the section
        :type section_name: :py:class:`str`
        :raises ValueError: If the structure of *dataset_config* is not
            valid.
        :return: The file dataset file names
        :rtype: :py:class:`tuple`
        """
        # Get the inputs file name
        file_name = self.__get_value(
            ConfigKeys.file, dataset_config, section_name=section_name,
            mandatory=True)

        # Get the inputs file name
        output_file_name = self.__get_value(
            ConfigKeys.output_file, dataset_config,
            section_name=section_name)

        if output_file_name is not None:
            return (file_name, output_file_name)
        else:
            return (file_name,)

    def __build_classifier(self, classifier_config, section_name):
        """Build a classifier.

        :param classifier_config: Section where the classifier is defined
        :type classifier_config: :py:class:`dict`
        :param section_name: Name of the section
        :type section_name: :py:class:`str`
        :return: The classifier object
        :rtype: Any subclass of :py:class:`~sklearn.base.ClassifierMixin`
        """
        # Get the classifier class
        classifier_cls = self.__get_value(
            ConfigKeys.classifier_cls, classifier_config,
            section_name=section_name, mandatory=True)

        # Get the classifier parameters
        classifier_params = self.__get_value(
            ConfigSections.parameters, classifier_config,
            section_name=section_name, is_section=True, mandatory=False)

        return classifier_cls(**classifier_params)

    def __build_fitness(self, fitness_config, section_name):
        """Build a fitness object.

        :param fitness_config: Section where the fitness is defined
        :type fitness_config: :py:class:`dict`
        :param section_name: Name of the section
        :type section_name: :py:class:`str`
        :return: The fitness object
        :rtype: Any subclass of :py:class:`~base.fitness.Fitness`
        """
        # Get the fitness class
        fitness_cls = self.__get_value(
            ConfigKeys.fitness_cls, fitness_config,
            section_name=section_name, mandatory=True)

        # Get the fitness parameters
        fitness_params = self.__get_value(
            ConfigSections.parameters, fitness_config,
            section_name=section_name, is_section=True)

        # Get the classifier configuration
        classifier_config = self.__get_value(
            ConfigSections.classifier, fitness_config,
            section_name=section_name, is_section=True, mandatory=True)

        # Build the classifier
        classifier = self.__build_classifier(
            classifier_config, section_name=section_name + self._toml_sep +
            ConfigSections.classifier)

        # Add the classifier to the fitness parameters
        fitness_params[ConfigSections.classifier] = classifier

        return fitness_cls(**fitness_params)
