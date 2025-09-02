# !/usr/bin/env python3
# -*- coding: utf-8 -*-

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

"""Unit test for :py:class:`~culebra.trainer.ea.abc.HeterogeneousEA`."""

import unittest

from culebra.trainer.topology import ring_destinations
from culebra.trainer.ea.abc import SinglePopEA, HeterogeneousEA
from culebra.solution.feature_selection import (
    Species,
    BitVector as Individual
)
from culebra.fitness_function.feature_selection import (
    KappaIndex,
    NumFeats,
    FSMultiObjectiveDatasetScorer
)
from culebra.tools import Dataset


# Fitness function
def KappaNumFeats(
    training_data,
    test_data=None,
    test_prop=None,
    cv_folds=None,
    classifier=None
):
    """Fitness Function."""
    return FSMultiObjectiveDatasetScorer(
        KappaIndex(
            training_data=training_data,
            test_data=test_data,
            test_prop=test_prop,
            cv_folds=cv_folds,
            classifier=classifier
        ),
        NumFeats()
    )


# Dataset
dataset = Dataset.load_from_uci(name="Wine")

# Preprocess the dataset
dataset = dataset.drop_missing().scale().remove_outliers(random_seed=0)


class MyTrainer(HeterogeneousEA):
    """Dummy implementation of an island-based evolutionary algorithm."""

    solution_cls = Individual
    species = Species(dataset.num_feats)

    _subpop_properties_mapping = {
        "pop_sizes": "pop_size",
        "crossover_funcs": "crossover_func",
        "mutation_funcs": "mutation_func",
        "crossover_probs": "crossover_prob",
        "mutation_probs": "mutation_prob",
        "gene_ind_mutation_probs": "gene_ind_mutation_prob",
        "selection_funcs": "selection_func",
        "selection_funcs_params": "selection_func_params"
    }

    def _search(self):
        pass

    def _generate_subtrainers(self):
        """Generate the subpopulation trainers."""

        def subtrainers_properties():
            """Obtain the properties of each subpopulation trainer.

            :raises RuntimeError: If the length of any properties sequence
                does not match the number of subpopulations.

            :return: The properties of each subpopulation trainer.
            :rtype: :py:class:`list`
            """
            # Get the common attributes from the container trainer
            cls = self.subtrainer_cls
            common_properties = {
                key: getattr(self, key)
                for key in cls.__init__.__code__.co_varnames
                if hasattr(self, key) and getattr(self, key) is not None
            }

            # Append subpopulation trainer custom atributes
            common_properties.update(self.subtrainer_params)

            # List with the common properties. Equal for all the subpopulations
            properties = []
            for _ in range(self.num_subtrainers):
                subpop_properties = {}
                for key, value in common_properties.items():
                    subpop_properties[key] = value
                properties.append(subpop_properties)

            # Particular properties for each subpop
            cls = self.__class__
            for (
                    property_sequence_name,
                    subpop_property_name
            ) in self._subpop_properties_mapping.items():

                # Values of the sequence
                property_sequence_values = getattr(
                    cls, property_sequence_name
                ).fget(self)

                # Check the properties' length
                if len(property_sequence_values) != self.num_subtrainers:
                    raise RuntimeError(
                        f"The length of {property_sequence_name} does not "
                        "match the number of subpopulations"
                    )
                for (
                        subpop_properties, subpop_property_value
                ) in zip(properties, property_sequence_values):
                    subpop_properties[
                        subpop_property_name] = subpop_property_value

            return properties

        # Get the subpopulations properties
        properties = subtrainers_properties()

        # Generate the subpopulations
        self._subtrainers = []

        for (
            index, (
                checkpoint_filename,
                subpop_properties
            )
        ) in enumerate(
            zip(self.subtrainer_checkpoint_filenames, properties)
        ):
            subtrainer = self.subtrainer_cls(**subpop_properties)
            subtrainer.checkpoint_filename = checkpoint_filename
            subtrainer.index = index
            subtrainer.container = self
            subtrainer.__class__._preprocess_iteration = (
                self.receive_representatives
            )
            subtrainer.__class__._postprocess_iteration = (
                self.send_representatives
            )
            self._subtrainers.append(subtrainer)

    @property
    def representation_topology_func(self):
        """Get and set the representation topology function."""
        return ring_destinations

    @property
    def representation_topology_func_params(self):
        """Get and set the representation topology function parameters."""
        return {}


class TrainerTester(unittest.TestCase):
    """Test :py:class:`~culebra.trainer.ea.abc.HeterogeneousEA`."""

    def test_init(self):
        """Test :py:meth:`~culebra.trainer.ea.abc.HeterogeneousEA.__init__`."""
        valid_fitness_func = KappaNumFeats(dataset)
        valid_subtrainer_cls = SinglePopEA
        valid_num_subtrainers = 3

        invalid_pop_size_types = (type, {}, 1.5)
        invalid_pop_size_values = (-1, 0)
        valid_pop_size = 13
        valid_pop_sizes = tuple(
            valid_pop_size + i for i in range(valid_num_subtrainers)
        )

        invalid_funcs = (1, 1.5, {})
        valid_func = len
        valid_funcs = (
            Individual.crossover1p,
            Individual.crossover2p,
            Individual.mutate
        )

        invalid_prob_types = (type, {}, len)
        invalid_prob_values = (-1, 2)
        valid_prob = 0.33
        valid_probs = tuple(
            valid_prob + i * 0.1 for i in range(valid_num_subtrainers)
        )

        invalid_params = (1, 1.5, Individual)
        valid_params = {"parameter": 12}
        valid_funcs_params = (
            {"parameter0": 12},
            {"parameter1": 13},
            {"parameter2": 14}
        )

        # Try invalid types for pop_sizes. Should fail
        for pop_size in invalid_pop_size_types:
            with self.assertRaises(TypeError):
                MyTrainer(
                    valid_fitness_func,
                    valid_subtrainer_cls,
                    num_subtrainers=valid_num_subtrainers,
                    pop_sizes=pop_size
                )

        # Try invalid values for pop_size. Should fail
        for pop_size in invalid_pop_size_values:
            with self.assertRaises(ValueError):
                MyTrainer(
                    valid_fitness_func,
                    valid_subtrainer_cls,
                    num_subtrainers=valid_num_subtrainers,
                    pop_sizes=pop_size
                )

        # Try a fixed value for pop_sizes,
        # all islands should have the same value
        trainer = MyTrainer(
            valid_fitness_func,
            valid_subtrainer_cls,
            num_subtrainers=valid_num_subtrainers,
            pop_sizes=valid_pop_size
        )

        # Check the length of the sequence
        self.assertEqual(len(trainer.pop_sizes), trainer.num_subtrainers)

        # Check that all the values match
        for island_pop_size in trainer.pop_sizes:
            self.assertEqual(island_pop_size, valid_pop_size)

        # Try different values of pop_size for each island
        trainer = MyTrainer(
            valid_fitness_func,
            valid_subtrainer_cls,
            num_subtrainers=valid_num_subtrainers,
            pop_sizes=valid_pop_sizes
        )
        for pop_size1, pop_size2 in zip(trainer.pop_sizes, valid_pop_sizes):
            self.assertEqual(pop_size1, pop_size2)

        # Try invalid types for crossover_funcs. Should fail
        for func in invalid_funcs:
            with self.assertRaises(TypeError):
                MyTrainer(
                    valid_fitness_func,
                    valid_subtrainer_cls,
                    num_subtrainers=valid_num_subtrainers,
                    crossover_funcs=func
                )

        # Try a fixed value for all the crossover functions,
        # all islands should have the same value
        trainer = MyTrainer(
            valid_fitness_func,
            valid_subtrainer_cls,
            num_subtrainers=valid_num_subtrainers,
            crossover_funcs=valid_func
        )
        # Check the length of the sequence
        self.assertEqual(len(trainer.crossover_funcs), trainer.num_subtrainers)

        # Check that all the values match
        for island_crossover_func in trainer.crossover_funcs:
            self.assertEqual(island_crossover_func, valid_func)

        # Try different values of crossover_func for each island
        trainer = MyTrainer(
            valid_fitness_func,
            valid_subtrainer_cls,
            num_subtrainers=valid_num_subtrainers,
            crossover_funcs=valid_funcs
        )
        for crossover_func1, crossover_func2 in zip(
            trainer.crossover_funcs, valid_funcs
        ):
            self.assertEqual(crossover_func1, crossover_func2)

        # Try invalid types for mutation_funcs. Should fail
        for func in invalid_funcs:
            with self.assertRaises(TypeError):
                MyTrainer(
                    valid_fitness_func,
                    valid_subtrainer_cls,
                    num_subtrainers=valid_num_subtrainers,
                    mutation_funcs=func
                )

        # Try a fixed value for all the mutation functions,
        # all islands should have the same value
        trainer = MyTrainer(
            valid_fitness_func,
            valid_subtrainer_cls,
            num_subtrainers=valid_num_subtrainers,
            mutation_funcs=valid_func
        )
        # Check the length of the sequence
        self.assertEqual(len(trainer.mutation_funcs), trainer.num_subtrainers)

        # Check that all the values match
        for island_mutation_func in trainer.mutation_funcs:
            self.assertEqual(island_mutation_func, valid_func)

        # Try different values of mutation_func for each island
        trainer = MyTrainer(
            valid_fitness_func,
            valid_subtrainer_cls,
            num_subtrainers=valid_num_subtrainers,
            mutation_funcs=valid_funcs
        )
        for mutation_func1, mutation_func2 in zip(
            trainer.mutation_funcs, valid_funcs
        ):
            self.assertEqual(mutation_func1, mutation_func2)

        # Try invalid types for selection_funcs. Should fail
        for func in invalid_funcs:
            with self.assertRaises(TypeError):
                MyTrainer(
                    valid_fitness_func,
                    valid_subtrainer_cls,
                    num_subtrainers=valid_num_subtrainers,
                    selection_funcs=func
                )

        # Try a fixed value for all the selection functions,
        # all islands should have the same value
        trainer = MyTrainer(
            valid_fitness_func,
            valid_subtrainer_cls,
            num_subtrainers=valid_num_subtrainers,
            selection_funcs=valid_func
        )
        # Check the length of the sequence
        self.assertEqual(len(trainer.selection_funcs), trainer.num_subtrainers)

        # Check that all the values match
        for island_selection_func in trainer.selection_funcs:
            self.assertEqual(island_selection_func, valid_func)

        # Try different values of selection_func for each island
        trainer = MyTrainer(
            valid_fitness_func,
            valid_subtrainer_cls,
            num_subtrainers=valid_num_subtrainers,
            selection_funcs=valid_funcs
        )
        for selection_func1, selection_func2 in zip(
            trainer.selection_funcs, valid_funcs
        ):
            self.assertEqual(selection_func1, selection_func2)

        # Try invalid types for crossover_probs. Should fail
        for prob in invalid_prob_types:
            with self.assertRaises(TypeError):
                MyTrainer(
                    valid_fitness_func,
                    valid_subtrainer_cls,
                    num_subtrainers=valid_num_subtrainers,
                    crossover_probs=prob
                )

        # Try invalid values for crossover_probs. Should fail
        for prob in invalid_prob_values:
            with self.assertRaises(ValueError):
                MyTrainer(
                    valid_fitness_func,
                    valid_subtrainer_cls,
                    num_subtrainers=valid_num_subtrainers,
                    crossover_probs=prob
                )

        # Try a fixed value for the crossover probability,
        # all islands should have the same value
        trainer = MyTrainer(
            valid_fitness_func,
            valid_subtrainer_cls,
            num_subtrainers=valid_num_subtrainers,
            crossover_probs=valid_prob
        )
        # Check the length of the sequence
        self.assertEqual(len(trainer.crossover_probs), trainer.num_subtrainers)

        # Check that all the values match
        for island_crossover_prob in trainer.crossover_probs:
            self.assertEqual(island_crossover_prob, valid_prob)

        # Try different values of crossover_prob for each island
        trainer = MyTrainer(
            valid_fitness_func,
            valid_subtrainer_cls,
            num_subtrainers=valid_num_subtrainers,
            crossover_probs=valid_probs
        )
        for prob1, prob2 in zip(trainer.crossover_probs, valid_probs):
            self.assertEqual(prob1, prob2)

        # Try invalid types for mutation_probs. Should fail
        for prob in invalid_prob_types:
            with self.assertRaises(TypeError):
                MyTrainer(
                    valid_fitness_func,
                    valid_subtrainer_cls,
                    num_subtrainers=valid_num_subtrainers,
                    mutation_probs=prob
                )

        # Try invalid values for mutation_probs. Should fail
        for prob in invalid_prob_values:
            with self.assertRaises(ValueError):
                MyTrainer(
                    valid_fitness_func,
                    valid_subtrainer_cls,
                    num_subtrainers=valid_num_subtrainers,
                    mutation_probs=prob
                )

        # Try a fixed value for the mutation probability,
        # all islands should have the same value
        trainer = MyTrainer(
            valid_fitness_func,
            valid_subtrainer_cls,
            num_subtrainers=valid_num_subtrainers,
            mutation_probs=valid_prob
        )
        # Check the length of the sequence
        self.assertEqual(len(trainer.mutation_probs), trainer.num_subtrainers)

        # Check that all the values match
        for island_mutation_prob in trainer.mutation_probs:
            self.assertEqual(island_mutation_prob, valid_prob)

        # Try different values of mutation_prob for each island
        trainer = MyTrainer(
            valid_fitness_func,
            valid_subtrainer_cls,
            num_subtrainers=valid_num_subtrainers,
            mutation_probs=valid_probs
        )
        for prob1, prob2 in zip(trainer.mutation_probs, valid_probs):
            self.assertEqual(prob1, prob2)

        # Try invalid types for gene_ind_mutation_probs. Should fail
        for prob in invalid_prob_types:
            with self.assertRaises(TypeError):
                MyTrainer(
                    valid_fitness_func,
                    valid_subtrainer_cls,
                    num_subtrainers=valid_num_subtrainers,
                    gene_ind_mutation_probs=prob
                )

        # Try invalid values for gene_ind_mutation_probs. Should fail
        for prob in invalid_prob_values:
            with self.assertRaises(ValueError):
                MyTrainer(
                    valid_fitness_func,
                    valid_subtrainer_cls,
                    num_subtrainers=valid_num_subtrainers,
                    gene_ind_mutation_probs=prob
                )

        # Try a fixed value for the mutation probability,
        # all islands should have the same value
        trainer = MyTrainer(
            valid_fitness_func,
            valid_subtrainer_cls,
            num_subtrainers=valid_num_subtrainers,
            gene_ind_mutation_probs=valid_prob
        )
        # Check the length of the sequence
        self.assertEqual(
            len(trainer.gene_ind_mutation_probs), trainer.num_subtrainers
        )

        # Check that all the values match
        for island_gene_ind_mutation_prob in trainer.gene_ind_mutation_probs:
            self.assertEqual(island_gene_ind_mutation_prob, valid_prob)

        # Try different values of gene_ind_mutation_prob for each island
        trainer = MyTrainer(
            valid_fitness_func,
            valid_subtrainer_cls,
            num_subtrainers=valid_num_subtrainers,
            gene_ind_mutation_probs=valid_probs
        )
        for prob1, prob2 in zip(trainer.gene_ind_mutation_probs, valid_probs):
            self.assertEqual(prob1, prob2)

        # Try invalid types for selection_funcs_params. Should fail
        for params in invalid_params:
            with self.assertRaises(TypeError):
                MyTrainer(
                    valid_fitness_func,
                    valid_subtrainer_cls,
                    num_subtrainers=valid_num_subtrainers,
                    selection_funcs_params=params
                )

        # Try a fixed value for the selection function parameters,
        # all islands should have the same value
        trainer = MyTrainer(
            valid_fitness_func,
            valid_subtrainer_cls,
            num_subtrainers=valid_num_subtrainers,
            selection_funcs_params=valid_params
        )
        # Check the length of the sequence
        self.assertEqual(
            len(trainer.selection_funcs_params), trainer.num_subtrainers
        )

        # Check that all the values match
        for island_selection_func_params in trainer.selection_funcs_params:
            self.assertEqual(island_selection_func_params, valid_params)

        # Try different values of selection_funcs_params for each island
        trainer = MyTrainer(
            valid_fitness_func,
            valid_subtrainer_cls,
            num_subtrainers=valid_num_subtrainers,
            selection_funcs_params=valid_funcs_params
        )
        for selection_func_params1, selection_func_params2 in zip(
            trainer.selection_funcs_params, valid_funcs_params
        ):
            self.assertEqual(selection_func_params1, selection_func_params2)

        # Test default params
        trainer = MyTrainer(
            valid_fitness_func,
            valid_subtrainer_cls,
            num_subtrainers=valid_num_subtrainers
        )

        # Default values for not initialized subpopulations should be None
        for pop_size in trainer.pop_sizes:
            self.assertEqual(pop_size, None)

        for crossover_func in trainer.crossover_funcs:
            self.assertEqual(crossover_func, None)

        for mutation_func in trainer.mutation_funcs:
            self.assertEqual(mutation_func, None)

        for selection_func in trainer.selection_funcs:
            self.assertEqual(selection_func, None)

        for crossover_prob in trainer.crossover_probs:
            self.assertEqual(crossover_prob, None)

        for mutation_prob in trainer.mutation_probs:
            self.assertEqual(mutation_prob, None)

        for gene_ind_mutation_prob in trainer.gene_ind_mutation_probs:
            self.assertEqual(gene_ind_mutation_prob, None)

        for selection_func_params in trainer.selection_funcs_params:
            self.assertEqual(selection_func_params, None)

    def test_repr(self):
        """Test the repr and str dunder methods."""
        # Set custom params
        fitness_func = KappaNumFeats(dataset)
        subtrainer_cls = SinglePopEA
        num_subtrainers = 3
        params = {
            "fitness_function": fitness_func,
            "subtrainer_cls": subtrainer_cls,
            "num_subtrainers": num_subtrainers
        }

        # Construct a parameterized trainer
        trainer = MyTrainer(**params)
        trainer._init_search()
        self.assertIsInstance(repr(trainer), str)
        self.assertIsInstance(str(trainer), str)


if __name__ == '__main__':
    unittest.main()
