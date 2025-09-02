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

:py:class:`culebra.trainer.abc.ParallelDistributedTrainer` class
================================================================

.. autoclass:: culebra.trainer.abc.ParallelDistributedTrainer

Class attributes
----------------
.. autoattribute:: culebra.trainer.abc.ParallelDistributedTrainer.stats_names
.. autoattribute:: culebra.trainer.abc.ParallelDistributedTrainer.objective_stats

Class methods
-------------
.. automethod:: culebra.trainer.abc.ParallelDistributedTrainer.load

Properties
----------
.. autoproperty:: culebra.trainer.abc.ParallelDistributedTrainer.fitness_function
.. autoproperty:: culebra.trainer.abc.ParallelDistributedTrainer.subtrainer_cls
.. autoproperty:: culebra.trainer.abc.ParallelDistributedTrainer.max_num_iters
.. autoproperty:: culebra.trainer.abc.ParallelDistributedTrainer.current_iter
.. autoproperty:: culebra.trainer.abc.ParallelDistributedTrainer.custom_termination_func
.. autoproperty:: culebra.trainer.abc.ParallelDistributedTrainer.num_subtrainers
.. autoproperty:: culebra.trainer.abc.ParallelDistributedTrainer.representation_size
.. autoproperty:: culebra.trainer.abc.ParallelDistributedTrainer.representation_freq
.. autoproperty:: culebra.trainer.abc.ParallelDistributedTrainer.representation_topology_func
.. autoproperty:: culebra.trainer.abc.ParallelDistributedTrainer.representation_topology_func_params
.. autoproperty:: culebra.trainer.abc.ParallelDistributedTrainer.representation_selection_func
.. autoproperty:: culebra.trainer.abc.ParallelDistributedTrainer.representation_selection_func_params
.. autoproperty:: culebra.trainer.abc.ParallelDistributedTrainer.checkpoint_enable
.. autoproperty:: culebra.trainer.abc.ParallelDistributedTrainer.checkpoint_freq
.. autoproperty:: culebra.trainer.abc.ParallelDistributedTrainer.checkpoint_filename
.. autoproperty:: culebra.trainer.abc.ParallelDistributedTrainer.verbose
.. autoproperty:: culebra.trainer.abc.ParallelDistributedTrainer.random_seed
.. autoproperty:: culebra.trainer.abc.ParallelDistributedTrainer.logbook
.. autoproperty:: culebra.trainer.abc.ParallelDistributedTrainer.num_evals
.. autoproperty:: culebra.trainer.abc.ParallelDistributedTrainer.runtime
.. autoproperty:: culebra.trainer.abc.ParallelDistributedTrainer.index
.. autoproperty:: culebra.trainer.abc.ParallelDistributedTrainer.container
.. autoproperty:: culebra.trainer.abc.ParallelDistributedTrainer.representatives
.. autoproperty:: culebra.trainer.abc.ParallelDistributedTrainer.subtrainer_params
.. autoproperty:: culebra.trainer.abc.ParallelDistributedTrainer.subtrainer_checkpoint_filenames
.. autoproperty:: culebra.trainer.abc.ParallelDistributedTrainer.subtrainers


Private properties
------------------
.. autoproperty:: culebra.trainer.abc.ParallelDistributedTrainer._subtrainer_suffixes

Static methods
--------------
.. automethod:: culebra.trainer.abc.ParallelDistributedTrainer.receive_representatives
.. automethod:: culebra.trainer.abc.ParallelDistributedTrainer.send_representatives

Methods
-------
.. automethod:: culebra.trainer.abc.ParallelDistributedTrainer.dump
.. automethod:: culebra.trainer.abc.ParallelDistributedTrainer.reset
.. automethod:: culebra.trainer.abc.ParallelDistributedTrainer.evaluate
.. automethod:: culebra.trainer.abc.ParallelDistributedTrainer.best_solutions
.. automethod:: culebra.trainer.abc.ParallelDistributedTrainer.best_representatives
.. automethod:: culebra.trainer.abc.ParallelDistributedTrainer.train
.. automethod:: culebra.trainer.abc.ParallelDistributedTrainer.test

Private methods
---------------
.. automethod:: culebra.trainer.abc.ParallelDistributedTrainer._generate_subtrainers
.. automethod:: culebra.trainer.abc.ParallelDistributedTrainer._get_state
.. automethod:: culebra.trainer.abc.ParallelDistributedTrainer._set_state
.. automethod:: culebra.trainer.abc.ParallelDistributedTrainer._save_state
.. automethod:: culebra.trainer.abc.ParallelDistributedTrainer._load_state
.. automethod:: culebra.trainer.abc.ParallelDistributedTrainer._new_state
.. automethod:: culebra.trainer.abc.ParallelDistributedTrainer._init_state
.. automethod:: culebra.trainer.abc.ParallelDistributedTrainer._reset_state
.. automethod:: culebra.trainer.abc.ParallelDistributedTrainer._init_internals
.. automethod:: culebra.trainer.abc.ParallelDistributedTrainer._reset_internals
.. automethod:: culebra.trainer.abc.ParallelDistributedTrainer._init_search
.. automethod:: culebra.trainer.abc.ParallelDistributedTrainer._search
.. automethod:: culebra.trainer.abc.ParallelDistributedTrainer._finish_search
.. automethod:: culebra.trainer.abc.ParallelDistributedTrainer._start_iteration
.. automethod:: culebra.trainer.abc.ParallelDistributedTrainer._preprocess_iteration
.. automethod:: culebra.trainer.abc.ParallelDistributedTrainer._do_iteration
.. automethod:: culebra.trainer.abc.ParallelDistributedTrainer._postprocess_iteration
.. automethod:: culebra.trainer.abc.ParallelDistributedTrainer._finish_iteration
.. automethod:: culebra.trainer.abc.ParallelDistributedTrainer._do_iteration_stats
.. automethod:: culebra.trainer.abc.ParallelDistributedTrainer._default_termination_func
.. automethod:: culebra.trainer.abc.ParallelDistributedTrainer._termination_criterion
.. automethod:: culebra.trainer.abc.ParallelDistributedTrainer._init_representatives
