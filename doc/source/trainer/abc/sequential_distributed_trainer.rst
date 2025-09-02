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

:py:class:`culebra.trainer.abc.SequentialDistributedTrainer` class
==================================================================

.. autoclass:: culebra.trainer.abc.SequentialDistributedTrainer

Class attributes
----------------
.. autoattribute:: culebra.trainer.abc.SequentialDistributedTrainer.stats_names
.. autoattribute:: culebra.trainer.abc.SequentialDistributedTrainer.objective_stats

Class methods
-------------
.. automethod:: culebra.trainer.abc.SequentialDistributedTrainer.load

Properties
----------
.. autoproperty:: culebra.trainer.abc.SequentialDistributedTrainer.fitness_function
.. autoproperty:: culebra.trainer.abc.SequentialDistributedTrainer.subtrainer_cls
.. autoproperty:: culebra.trainer.abc.SequentialDistributedTrainer.max_num_iters
.. autoproperty:: culebra.trainer.abc.SequentialDistributedTrainer.current_iter
.. autoproperty:: culebra.trainer.abc.SequentialDistributedTrainer.custom_termination_func
.. autoproperty:: culebra.trainer.abc.SequentialDistributedTrainer.num_subtrainers
.. autoproperty:: culebra.trainer.abc.SequentialDistributedTrainer.representation_size
.. autoproperty:: culebra.trainer.abc.SequentialDistributedTrainer.representation_freq
.. autoproperty:: culebra.trainer.abc.SequentialDistributedTrainer.representation_topology_func
.. autoproperty:: culebra.trainer.abc.SequentialDistributedTrainer.representation_topology_func_params
.. autoproperty:: culebra.trainer.abc.SequentialDistributedTrainer.representation_selection_func
.. autoproperty:: culebra.trainer.abc.SequentialDistributedTrainer.representation_selection_func_params
.. autoproperty:: culebra.trainer.abc.SequentialDistributedTrainer.checkpoint_enable
.. autoproperty:: culebra.trainer.abc.SequentialDistributedTrainer.checkpoint_freq
.. autoproperty:: culebra.trainer.abc.SequentialDistributedTrainer.checkpoint_filename
.. autoproperty:: culebra.trainer.abc.SequentialDistributedTrainer.verbose
.. autoproperty:: culebra.trainer.abc.SequentialDistributedTrainer.random_seed
.. autoproperty:: culebra.trainer.abc.SequentialDistributedTrainer.logbook
.. autoproperty:: culebra.trainer.abc.SequentialDistributedTrainer.num_evals
.. autoproperty:: culebra.trainer.abc.SequentialDistributedTrainer.runtime
.. autoproperty:: culebra.trainer.abc.SequentialDistributedTrainer.index
.. autoproperty:: culebra.trainer.abc.SequentialDistributedTrainer.container
.. autoproperty:: culebra.trainer.abc.SequentialDistributedTrainer.representatives
.. autoproperty:: culebra.trainer.abc.SequentialDistributedTrainer.subtrainer_params
.. autoproperty:: culebra.trainer.abc.SequentialDistributedTrainer.subtrainer_checkpoint_filenames
.. autoproperty:: culebra.trainer.abc.SequentialDistributedTrainer.subtrainers

Private properties
------------------
.. autoproperty:: culebra.trainer.abc.SequentialDistributedTrainer._subtrainer_suffixes

Static methods
--------------
.. automethod:: culebra.trainer.abc.SequentialDistributedTrainer.receive_representatives
.. automethod:: culebra.trainer.abc.SequentialDistributedTrainer.send_representatives

Methods
-------
.. automethod:: culebra.trainer.abc.SequentialDistributedTrainer.dump
.. automethod:: culebra.trainer.abc.SequentialDistributedTrainer.reset
.. automethod:: culebra.trainer.abc.SequentialDistributedTrainer.evaluate
.. automethod:: culebra.trainer.abc.SequentialDistributedTrainer.best_solutions
.. automethod:: culebra.trainer.abc.SequentialDistributedTrainer.best_representatives
.. automethod:: culebra.trainer.abc.SequentialDistributedTrainer.train
.. automethod:: culebra.trainer.abc.SequentialDistributedTrainer.test

Private methods
---------------
.. automethod:: culebra.trainer.abc.SequentialDistributedTrainer._generate_subtrainers
.. automethod:: culebra.trainer.abc.SequentialDistributedTrainer._get_state
.. automethod:: culebra.trainer.abc.SequentialDistributedTrainer._set_state
.. automethod:: culebra.trainer.abc.SequentialDistributedTrainer._save_state
.. automethod:: culebra.trainer.abc.SequentialDistributedTrainer._load_state
.. automethod:: culebra.trainer.abc.SequentialDistributedTrainer._new_state
.. automethod:: culebra.trainer.abc.SequentialDistributedTrainer._init_state
.. automethod:: culebra.trainer.abc.SequentialDistributedTrainer._reset_state
.. automethod:: culebra.trainer.abc.SequentialDistributedTrainer._init_internals
.. automethod:: culebra.trainer.abc.SequentialDistributedTrainer._reset_internals
.. automethod:: culebra.trainer.abc.SequentialDistributedTrainer._init_search
.. automethod:: culebra.trainer.abc.SequentialDistributedTrainer._search
.. automethod:: culebra.trainer.abc.SequentialDistributedTrainer._finish_search
.. automethod:: culebra.trainer.abc.SequentialDistributedTrainer._start_iteration
.. automethod:: culebra.trainer.abc.SequentialDistributedTrainer._preprocess_iteration
.. automethod:: culebra.trainer.abc.SequentialDistributedTrainer._do_iteration
.. automethod:: culebra.trainer.abc.SequentialDistributedTrainer._postprocess_iteration
.. automethod:: culebra.trainer.abc.SequentialDistributedTrainer._finish_iteration
.. automethod:: culebra.trainer.abc.SequentialDistributedTrainer._do_iteration_stats
.. automethod:: culebra.trainer.abc.SequentialDistributedTrainer._default_termination_func
.. automethod:: culebra.trainer.abc.SequentialDistributedTrainer._termination_criterion
.. automethod:: culebra.trainer.abc.SequentialDistributedTrainer._init_representatives
