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

:py:class:`culebra.trainer.abc.IslandsTrainer` class
====================================================

.. autoclass:: culebra.trainer.abc.IslandsTrainer

Class attributes
----------------
.. autoattribute:: culebra.trainer.abc.IslandsTrainer.stats_names
.. autoattribute:: culebra.trainer.abc.IslandsTrainer.objective_stats

Class methods
-------------
.. automethod:: culebra.trainer.abc.IslandsTrainer.load

Properties
----------
.. autoproperty:: culebra.trainer.abc.IslandsTrainer.solution_cls
.. autoproperty:: culebra.trainer.abc.IslandsTrainer.species
.. autoproperty:: culebra.trainer.abc.IslandsTrainer.fitness_function
.. autoproperty:: culebra.trainer.abc.IslandsTrainer.subtrainer_cls
.. autoproperty:: culebra.trainer.abc.IslandsTrainer.max_num_iters
.. autoproperty:: culebra.trainer.abc.IslandsTrainer.current_iter
.. autoproperty:: culebra.trainer.abc.IslandsTrainer.custom_termination_func
.. autoproperty:: culebra.trainer.abc.IslandsTrainer.num_subtrainers
.. autoproperty:: culebra.trainer.abc.IslandsTrainer.representation_size
.. autoproperty:: culebra.trainer.abc.IslandsTrainer.representation_freq
.. autoproperty:: culebra.trainer.abc.IslandsTrainer.representation_topology_func
.. autoproperty:: culebra.trainer.abc.IslandsTrainer.representation_topology_func_params
.. autoproperty:: culebra.trainer.abc.IslandsTrainer.representation_selection_func
.. autoproperty:: culebra.trainer.abc.IslandsTrainer.representation_selection_func_params
.. autoproperty:: culebra.trainer.abc.IslandsTrainer.checkpoint_enable
.. autoproperty:: culebra.trainer.abc.IslandsTrainer.checkpoint_freq
.. autoproperty:: culebra.trainer.abc.IslandsTrainer.checkpoint_filename
.. autoproperty:: culebra.trainer.abc.IslandsTrainer.verbose
.. autoproperty:: culebra.trainer.abc.IslandsTrainer.random_seed
.. autoproperty:: culebra.trainer.abc.IslandsTrainer.logbook
.. autoproperty:: culebra.trainer.abc.IslandsTrainer.num_evals
.. autoproperty:: culebra.trainer.abc.IslandsTrainer.runtime
.. autoproperty:: culebra.trainer.abc.IslandsTrainer.index
.. autoproperty:: culebra.trainer.abc.IslandsTrainer.container
.. autoproperty:: culebra.trainer.abc.IslandsTrainer.representatives
.. autoproperty:: culebra.trainer.abc.IslandsTrainer.subtrainer_params
.. autoproperty:: culebra.trainer.abc.IslandsTrainer.subtrainer_checkpoint_filenames
.. autoproperty:: culebra.trainer.abc.IslandsTrainer.subtrainers

Private properties
------------------
.. autoproperty:: culebra.trainer.abc.IslandsTrainer._subtrainer_suffixes

Static methods
--------------
.. automethod:: culebra.trainer.abc.IslandsTrainer.receive_representatives
.. automethod:: culebra.trainer.abc.IslandsTrainer.send_representatives

Methods
-------
.. automethod:: culebra.trainer.abc.IslandsTrainer.reset
.. automethod:: culebra.trainer.abc.IslandsTrainer.evaluate
.. automethod:: culebra.trainer.abc.IslandsTrainer.best_solutions
.. automethod:: culebra.trainer.abc.IslandsTrainer.best_representatives
.. automethod:: culebra.trainer.abc.IslandsTrainer.train
.. automethod:: culebra.trainer.abc.IslandsTrainer.test

Private methods
---------------
.. automethod:: culebra.trainer.abc.IslandsTrainer.dump
.. automethod:: culebra.trainer.abc.IslandsTrainer._generate_subtrainers
.. automethod:: culebra.trainer.abc.IslandsTrainer._get_state
.. automethod:: culebra.trainer.abc.IslandsTrainer._set_state
.. automethod:: culebra.trainer.abc.IslandsTrainer._save_state
.. automethod:: culebra.trainer.abc.IslandsTrainer._load_state
.. automethod:: culebra.trainer.abc.IslandsTrainer._new_state
.. automethod:: culebra.trainer.abc.IslandsTrainer._init_state
.. automethod:: culebra.trainer.abc.IslandsTrainer._reset_state
.. automethod:: culebra.trainer.abc.IslandsTrainer._init_internals
.. automethod:: culebra.trainer.abc.IslandsTrainer._reset_internals
.. automethod:: culebra.trainer.abc.IslandsTrainer._init_search
.. automethod:: culebra.trainer.abc.IslandsTrainer._search
.. automethod:: culebra.trainer.abc.IslandsTrainer._finish_search
.. automethod:: culebra.trainer.abc.IslandsTrainer._start_iteration
.. automethod:: culebra.trainer.abc.IslandsTrainer._preprocess_iteration
.. automethod:: culebra.trainer.abc.IslandsTrainer._do_iteration
.. automethod:: culebra.trainer.abc.IslandsTrainer._postprocess_iteration
.. automethod:: culebra.trainer.abc.IslandsTrainer._finish_iteration
.. automethod:: culebra.trainer.abc.IslandsTrainer._do_iteration_stats
.. automethod:: culebra.trainer.abc.IslandsTrainer._default_termination_func
.. automethod:: culebra.trainer.abc.IslandsTrainer._termination_criterion
.. automethod:: culebra.trainer.abc.IslandsTrainer._init_representatives
.. automethod:: culebra.trainer.abc.IslandsTrainer._set_cooperative_fitness
