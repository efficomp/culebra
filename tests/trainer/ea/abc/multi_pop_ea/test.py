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

"""Unit test for :py:class:`~culebra.trainer.ea.abc.MultiPopEA`."""

import unittest

from culebra.trainer.abc import SingleSpeciesTrainer
from culebra.trainer.topology import ring_destinations
from culebra.trainer.ea.abc import SinglePopEA, MultiPopEA
from culebra.solution.feature_selection import (
    Species,
    BitVector as Individual
)
from culebra.fitness_function.feature_selection import KappaNumFeats as Fitness
from culebra.tools import Dataset


# Dataset
dataset = Dataset.load_from_uci(name="Wine")

# Preprocess the dataset
dataset = dataset.drop_missing().scale().remove_outliers(random_seed=0)

# Default species for all the tests
species = Species(num_feats=dataset.num_feats)


class MyTrainer(MultiPopEA):
    """Dummy implementation of a distributed trainer."""

    solution_cls = Individual
    species = Species(dataset.num_feats)

    def _generate_subtrainers(self):
        """Generate the subpopulation trainers.

        Also assign an :py:attr:`~culebra.trainer.ea.abc.SinglePopEA.index`
        and a :py:attr:`~culebra.trainer.ea.abc.SinglePopEA.container` to each
        subpopulation :py:class:`~culebra.trainer.ea.abc.SinglePopEA` trainer,
        change the subpopulation trainers'
        :py:attr:`~culebra.trainer.ea.abc.SinglePopEA.checkpoint_filename`
        according to the container checkpointing file name and each
        subpopulation index.

        Finally, the
        :py:meth:`~culebra.trainer.ea.abc.SinglePopEA._preprocess_iteration`
        and
        :py:meth:`~culebra.trainer.ea.abc.SinglePopEA._postprocess_iteration`
        methods of the
        :py:attr:`~culebra.trainer.ea.abc.HomogeneousIslandsEA.subtrainer_cls`
        class are dynamically overridden, in order to allow individuals
        exchange between subpopulation trainers, if necessary
        """

        def subtrainers_properties():
            """Return the subpopulation trainers' properties."""
            # Get the attributes from the container trainer
            cls = self.subtrainer_cls
            properties = {
                key: getattr(self, key)
                for key in cls.__init__.__code__.co_varnames
                if hasattr(self, key) and getattr(self, key) is not None
            }

            # Append subpopulation trainer custom atributes
            properties.update(self.subtrainer_params)

            return properties

        # Get the subpopulations properties
        properties = subtrainers_properties()

        # Generate the subpopulations
        self._subtrainers = []

        for (
            index,
            checkpoint_filename
        ) in enumerate(self.subtrainer_checkpoint_filenames):
            subtrainer = self.subtrainer_cls(**properties)
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
    """Test :py:class:`~culebra.trainer.ea.abc.MultiPopEA`."""

    def test_subtrainer_cls(self):
        """Test the subtrainer_cls property."""
        valid_fitness_func = Fitness(dataset)
        valid_subtrainer_cls = SinglePopEA

        # Try invalid subtrainer_cls. Should fail
        invalid_trainer_classes = (
            tuple, str, None, 'a', 1, SingleSpeciesTrainer
        )
        for cls in invalid_trainer_classes:
            with self.assertRaises(TypeError):
                MyTrainer(
                    valid_fitness_func,
                    cls
                )

        # Test default params
        trainer = MyTrainer(valid_fitness_func, valid_subtrainer_cls)
        self.assertEqual(trainer.subtrainer_cls, valid_subtrainer_cls)

    def test_repr(self):
        """Test the repr and str dunder methods."""
        # Create a default trainer
        valid_fitness_func = Fitness(dataset)
        valid_subtrainer_cls = SinglePopEA
        trainer = MyTrainer(valid_fitness_func, valid_subtrainer_cls)
        trainer._init_search()
        self.assertIsInstance(repr(trainer), str)
        self.assertIsInstance(str(trainer), str)


if __name__ == '__main__':
    unittest.main()
