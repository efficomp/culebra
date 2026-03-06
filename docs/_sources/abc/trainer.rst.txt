..
   This file is part of

   Culebra is free software: you can redistribute it and/or modify it under the
   terms of the GNU General Public License as published by the Free Software
   Foundation, either version 3 of the License, or (at your option) any later
   version.

   Culebra is distributed in the hope that it will be useful, but WITHOUT ANY
   WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
   FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
   details.

   You should have received a copy of the GNU General Public License along with
   Culebra. If not, see <http://www.gnu.org/licenses/>.

   This work is supported by projects PGC2018-098813-B-C31 and
   PID2022-137461NB-C31, both funded by the Spanish "Ministerio de Ciencia,
   Innovación y Universidades" and by the European Regional Development Fund
   (ERDF).

:class:`culebra.abc.Trainer` class
==================================

.. autoclass:: culebra.abc.Trainer

Class methods
-------------
.. automethod:: culebra.abc.Trainer.load

Properties
----------
.. autoproperty:: culebra.abc.Trainer.cooperative_fitness_estimation_func
.. autoproperty:: culebra.abc.Trainer.fitness_func
.. autoproperty:: culebra.abc.Trainer.iteration_metric_names
.. autoproperty:: culebra.abc.Trainer.iteration_obj_stats
.. autoproperty:: culebra.abc.Trainer.logbook
.. autoproperty:: culebra.abc.Trainer.num_evals
.. autoproperty:: culebra.abc.Trainer.num_iters
.. autoproperty:: culebra.abc.Trainer.runtime
.. autoproperty:: culebra.abc.Trainer.training_finished

Private properties
------------------
.. autoproperty:: culebra.abc.Trainer._default_cooperative_fitness_estimation_func

Methods
-------
.. automethod:: culebra.abc.Trainer.best_cooperators
.. automethod:: culebra.abc.Trainer.best_solutions
.. automethod:: culebra.abc.Trainer.dump
.. automethod:: culebra.abc.Trainer.evaluate
.. automethod:: culebra.abc.Trainer.reset
.. automethod:: culebra.abc.Trainer.test
.. automethod:: culebra.abc.Trainer.train

Private methods
---------------
.. automethod:: culebra.abc.Trainer._do_training
.. automethod:: culebra.abc.Trainer._finish_training
.. automethod:: culebra.abc.Trainer._init_training
