#!/usr/bin/env python3
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

"""Unit test for :class:`~culebra.abc.Trainer`."""

import unittest

import numpy as np
from deap.tools import HallOfFame

from culebra.abc import (
    FitnessFunction,
    Solution,
    Species,
    Trainer
)


ITER_OBJ_STATS = {
    "Avg": np.mean,
    "Std": np.std,
    "Min": np.min,
    "Max": np.max,
}
"""Statistics calculated for each objective every iteration."""


class MySolution(Solution):
    """Dummy subclass to test the :class:`~culebra.abc.Solution` class."""

    def __init__(
        self,
        species,
        fitness_cls,
        val=0
    ) -> None:
        """Construct a default solution.

        :param species: The species the solution will belong to
        :type species: Any subclass of :class:`~culebra.abc.Species`
        :param fitness: The solution's fitness class
        :type fitness: Any subclass of :class:`~culebra.abc.Fitness`
        :param val: A value
        :type val: int
        :raises TypeError: If *species* is not a valid species
        :raises TypeError: If *fitness_cls* is not a valid fitness class
        """
        # Init the superclass
        super().__init__(species, fitness_cls)
        self.val = val

class MyOtherSolution(MySolution):
    """Dummy subclass to test the :class:`~culebra.abc.Solution` class."""


class MySpecies(Species):
    """Dummy subclass to test the :class:`~culebra.abc.Species` class."""

    def is_member(self, sol):
        """Check if a solution meets the constraints imposed by the species."""
        return True

class MyOtherSpecies(MySpecies):
    """Dummy subclass to test the :class:`~culebra.abc.Species` class."""


class MyFitnessFunction(FitnessFunction):
    """Dummy implementation of a fitness function."""

    @property
    def obj_weights(self):
        """Objective weights."""
        return (-1, 1)

    @property
    def obj_names(self):
        """Objective names."""
        return ("min", "max")

    def evaluate(self, sol, index=None, cooperators=None):
        """Evaluate one solution.

        Dummy implementation of the evaluation function.

        Obtain the minimum and maximum of the values stored by *sol* and
        *cooperators* (if provided)
        """
        min_val = max_val = sol.val
        if cooperators is not None:
            for other in cooperators:
                if other is not None:
                    min_val = min(other.val, min_val)
                    max_val = max(other.val, max_val)

        sol.fitness.values = (min_val, max_val)

        return sol.fitness


class MyOtherFitnessFunction(FitnessFunction):
    """Dummy implementation of a fitness function."""

    @property
    def obj_weights(self):
        """Objective weights."""
        return (-1, 1)

    @property
    def obj_names(self):
        """Objective names."""
        return ("doublemin", "doublemax",)

    def evaluate(self, sol, index=None, cooperators=None):
        """Evaluate one solution.

        Dummy implementation of the evaluation function.

        Obtain the double of the minimum and maximum of the values stored by
        *sol* and *cooperators* (if provided)
        """
        min_val = max_val = sol.val
        if cooperators is not None:
            for other in cooperators:
                if other is not None:
                    min_val = min(other.val, min_val)
                    max_val = max(other.val, max_val)

        sol.fitness.values = (min_val*2, max_val*2)

        return sol.fitness


class MyTrainer(Trainer):
    """Dummy implementation of a trainer method."""

    def __init__(self, fitness_func):
        super().__init__()
        self._fitness_func = fitness_func

    @property
    def fitness_func(self):
        return self._fitness_func

    @property
    def iteration_metric_names(self):
        return ('Iter', 'NEvals')

    @property
    def iteration_obj_stats(self):
        return ITER_OBJ_STATS

    @property
    def training_finished(self):
        return True

    @property
    def logbook(self):
        return None

    @property
    def num_evals(self):
        return 1

    @property
    def num_iters(self):
        return 1

    @property
    def runtime(self):
        return 1

    def reset(self):
        pass

    def _init_training(self):
        pass

    def _finish_training(self):
        pass

    def _do_training(self):
        pass

    def best_solutions(self):
        species = MySpecies()
        solution = MySolution(species, self.fitness_func.fitness_cls)
        self.evaluate(solution)
        population = (solution,)

        hof = HallOfFame(population)
        hof.update(population)
        return (hof,)


class TrainerTester(unittest.TestCase):
    """Test :class:`~culebra.abc.Trainer`."""

    def test_evaluate(self):
        """Test the solution evaluation."""
        # Create the species
        species = MySpecies()

        # Create the trainer
        trainer = MyTrainer(MyFitnessFunction())

        fitness_cls = trainer.fitness_func.fitness_cls

        # Create one solution
        sol = MySolution(species, fitness_cls, 2)


        # Omit the fitness function.
        # The default training funtion should be used
        num_evals = trainer.evaluate(sol)
        self.assertEqual(
            sol.fitness.values, (sol.val,) * sol.fitness.num_obj
        )
        self.assertEqual(
            sol.fitness.names,
            trainer.fitness_func.obj_names
        )
        self.assertEqual(num_evals, 1)

        # Provide a different fitness function.
        other_fitness_func = MyOtherFitnessFunction()
        num_evals = trainer.evaluate(sol, other_fitness_func)
        self.assertEqual(
            sol.fitness.values, (sol.val*2,) * sol.fitness.num_obj
        )
        self.assertEqual(
            sol.fitness.names,
            other_fitness_func.obj_names
        )
        self.assertEqual(num_evals, 1)

        # Provide cooperators
        coop1 = MySolution(species, fitness_cls, 1)
        coop2 = MySolution(species, fitness_cls, 3)
        cooperators = [[coop1], [coop2]]

        num_evals = trainer.evaluate(sol, cooperators=cooperators)
        self.assertTrue(
            (
                sol.fitness.values ==
                np.average(
                    ((coop1.val, coop2.val), (sol.val, sol.val)),
                    axis=0
                )
            ).all()
        )
        self.assertEqual(num_evals, len(cooperators))

        trainer.evaluate(sol, other_fitness_func, cooperators=cooperators)
        self.assertTrue(
            (
                sol.fitness.values ==
                np.average(
                    ((coop1.val, coop2.val), (sol.val, sol.val)),
                    axis=0
                ) * 2
            ).all()
        )
        self.assertEqual(num_evals, len(cooperators))

    def test_cooperative_fitness_estimation_func(self):
        """Test the cooperative_fitness_estimation_func property."""
        # Create the species
        species = MySpecies()

        # Create the trainer
        trainer = MyTrainer(MyFitnessFunction())
        fitness_cls = trainer.fitness_func.fitness_cls

        # Create one solution
        sol = MySolution(species, fitness_cls, 1)
        # Provide cooperators
        coop1 = MySolution(species, fitness_cls, 2)
        coop2 = MySolution(species, fitness_cls, 3)
        cooperators = [[coop1], [coop2]]

        # Set a new cooperative fitness estimation function
        trainer.cooperative_fitness_estimation_func = (
            lambda fitness_trials: np.max(fitness_trials, axis=0)
        )

        num_evals = trainer.evaluate(sol, cooperators=cooperators)
        self.assertEqual(
            sol.fitness.values,
            (sol.val, coop2.val)
        )
        self.assertEqual(num_evals, len(cooperators))

    def test_test(self):
        """Test :meth:`~culebra.abc.Trainer.test`."""
        # Create the trainer
        valid_fitness_func = MyOtherFitnessFunction()
        trainer = MyTrainer(valid_fitness_func)

        # Not a valid sequence of hofs
        hofs = None
        with self.assertRaises(TypeError):
            trainer.test(hofs, valid_fitness_func)

        # Not a valid sequence of hofs
        hofs = ["a"]
        with self.assertRaises(ValueError):
            trainer.test(hofs, valid_fitness_func)

        # Train
        trainer.train()
        hofs = trainer.best_solutions()

        # Not a valid fitness function
        with self.assertRaises(TypeError):
            trainer.test(hofs, fitness_func='a')

        # Not a valid sequence of cooperator solutions
        with self.assertRaises(TypeError):
            trainer.test(hofs, valid_fitness_func, cooperators=1)

        # Not a valid sequence of cooperator solutions
        with self.assertRaises(ValueError):
            trainer.test(hofs, valid_fitness_func, cooperators=['a'])

        # cooperators and hofs must have the same size
        # (the number of species)
        cooperators = (hofs[0][0],) * (len(hofs) + 1)
        with self.assertRaises(ValueError):
            trainer.test(
                hofs, valid_fitness_func, cooperators=cooperators
            )

        trainer.test(hofs, valid_fitness_func)
        # Check the test fitness values
        for hof in hofs:
            for sol in hof:
                self.assertEqual(
                    sol.fitness.values, (sol.val * 2,) * sol.fitness.num_obj
                )


    def test_repr(self):
        """Test the repr and str dunder methods."""
        trainer = MyTrainer(MyFitnessFunction())
        self.assertIsInstance(repr(trainer), str)
        self.assertIsInstance(str(trainer), str)


if __name__ == '__main__':
    unittest.main()
