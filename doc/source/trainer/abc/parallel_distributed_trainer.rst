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

:class:`culebra.trainer.abc.ParallelDistributedTrainer` class
=============================================================

.. autoclass:: culebra.trainer.abc.ParallelDistributedTrainer

Class methods
-------------
.. automethod:: culebra.trainer.abc.ParallelDistributedTrainer.load

Properties
----------
.. autoproperty:: culebra.trainer.abc.ParallelDistributedTrainer.cooperative_fitness_estimation_func
.. autoproperty:: culebra.trainer.abc.ParallelDistributedTrainer.fitness_func
.. autoproperty:: culebra.trainer.abc.ParallelDistributedTrainer.iteration_metric_names
.. autoproperty:: culebra.trainer.abc.ParallelDistributedTrainer.iteration_obj_stats
.. autoproperty:: culebra.trainer.abc.ParallelDistributedTrainer.logbook
.. autoproperty:: culebra.trainer.abc.ParallelDistributedTrainer.num_evals
.. autoproperty:: culebra.trainer.abc.ParallelDistributedTrainer.num_iters
.. autoproperty:: culebra.trainer.abc.ParallelDistributedTrainer.num_representatives
.. autoproperty:: culebra.trainer.abc.ParallelDistributedTrainer.num_subtrainers
.. autoproperty:: culebra.trainer.abc.ParallelDistributedTrainer.representatives_exchange_freq
.. autoproperty:: culebra.trainer.abc.ParallelDistributedTrainer.representatives_selection_func
.. autoproperty:: culebra.trainer.abc.ParallelDistributedTrainer.runtime
.. autoproperty:: culebra.trainer.abc.ParallelDistributedTrainer.subtrainers
.. autoproperty:: culebra.trainer.abc.ParallelDistributedTrainer.topology_func
.. autoproperty:: culebra.trainer.abc.ParallelDistributedTrainer.training_finished

Private properties
------------------
.. autoproperty:: culebra.trainer.abc.ParallelDistributedTrainer._default_cooperative_fitness_estimation_func
.. autoproperty:: culebra.trainer.abc.ParallelDistributedTrainer._default_num_representatives
.. autoproperty:: culebra.trainer.abc.ParallelDistributedTrainer._default_representatives_exchange_freq
.. autoproperty:: culebra.trainer.abc.ParallelDistributedTrainer._default_representatives_selection_func
.. autoproperty:: culebra.trainer.abc.ParallelDistributedTrainer._default_topology_func

Static methods
--------------
.. automethod:: culebra.trainer.abc.ParallelDistributedTrainer.receive_representatives
.. automethod:: culebra.trainer.abc.ParallelDistributedTrainer.send_representatives

Methods
-------
.. automethod:: culebra.trainer.abc.ParallelDistributedTrainer.best_cooperators
.. automethod:: culebra.trainer.abc.ParallelDistributedTrainer.best_solutions
.. automethod:: culebra.trainer.abc.ParallelDistributedTrainer.dump
.. automethod:: culebra.trainer.abc.ParallelDistributedTrainer.evaluate
.. automethod:: culebra.trainer.abc.ParallelDistributedTrainer.reset
.. automethod:: culebra.trainer.abc.ParallelDistributedTrainer.test
.. automethod:: culebra.trainer.abc.ParallelDistributedTrainer.train

Private methods
---------------
.. automethod:: culebra.trainer.abc.ParallelDistributedTrainer._do_training
.. automethod:: culebra.trainer.abc.ParallelDistributedTrainer._finish_training
.. automethod:: culebra.trainer.abc.ParallelDistributedTrainer._init_internals
.. automethod:: culebra.trainer.abc.ParallelDistributedTrainer._init_training
.. automethod:: culebra.trainer.abc.ParallelDistributedTrainer._reset_internals
