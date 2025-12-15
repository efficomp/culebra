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

:class:`culebra.trainer.abc.DistributedTrainer` class
=====================================================

.. autoclass:: culebra.trainer.abc.DistributedTrainer

Class attributes
----------------
.. autoattribute:: culebra.trainer.abc.DistributedTrainer.objective_stats
.. autoattribute:: culebra.trainer.abc.DistributedTrainer.stats_names

Class methods
-------------
.. automethod:: culebra.trainer.abc.DistributedTrainer.load

Properties
----------
.. autoproperty:: culebra.trainer.abc.DistributedTrainer.checkpoint_activation
.. autoproperty:: culebra.trainer.abc.DistributedTrainer.checkpoint_filename
.. autoproperty:: culebra.trainer.abc.DistributedTrainer.checkpoint_freq
.. autoproperty:: culebra.trainer.abc.DistributedTrainer.container
.. autoproperty:: culebra.trainer.abc.DistributedTrainer.current_iter
.. autoproperty:: culebra.trainer.abc.DistributedTrainer.custom_termination_func
.. autoproperty:: culebra.trainer.abc.DistributedTrainer.fitness_function
.. autoproperty:: culebra.trainer.abc.DistributedTrainer.index
.. autoproperty:: culebra.trainer.abc.DistributedTrainer.logbook
.. autoproperty:: culebra.trainer.abc.DistributedTrainer.max_num_iters
.. autoproperty:: culebra.trainer.abc.DistributedTrainer.num_evals
.. autoproperty:: culebra.trainer.abc.DistributedTrainer.num_subtrainers
.. autoproperty:: culebra.trainer.abc.DistributedTrainer.random_seed
.. autoproperty:: culebra.trainer.abc.DistributedTrainer.representation_freq
.. autoproperty:: culebra.trainer.abc.DistributedTrainer.representation_selection_func
.. autoproperty:: culebra.trainer.abc.DistributedTrainer.representation_selection_func_params
.. autoproperty:: culebra.trainer.abc.DistributedTrainer.representation_size
.. autoproperty:: culebra.trainer.abc.DistributedTrainer.representation_topology_func
.. autoproperty:: culebra.trainer.abc.DistributedTrainer.representation_topology_func_params
.. autoproperty:: culebra.trainer.abc.DistributedTrainer.representatives
.. autoproperty:: culebra.trainer.abc.DistributedTrainer.runtime
.. autoproperty:: culebra.trainer.abc.DistributedTrainer.subtrainer_checkpoint_filenames
.. autoproperty:: culebra.trainer.abc.DistributedTrainer.subtrainer_cls
.. autoproperty:: culebra.trainer.abc.DistributedTrainer.subtrainer_params
.. autoproperty:: culebra.trainer.abc.DistributedTrainer.subtrainers
.. autoproperty:: culebra.trainer.abc.DistributedTrainer.verbosity

Private properties
------------------
.. autoproperty:: culebra.trainer.abc.DistributedTrainer._default_checkpoint_activation
.. autoproperty:: culebra.trainer.abc.DistributedTrainer._default_checkpoint_filename
.. autoproperty:: culebra.trainer.abc.DistributedTrainer._default_checkpoint_freq
.. autoproperty:: culebra.trainer.abc.DistributedTrainer._default_index
.. autoproperty:: culebra.trainer.abc.DistributedTrainer._default_max_num_iters
.. autoproperty:: culebra.trainer.abc.DistributedTrainer._default_num_subtrainers
.. autoproperty:: culebra.trainer.abc.DistributedTrainer._default_representation_freq
.. autoproperty:: culebra.trainer.abc.DistributedTrainer._default_representation_selection_func
.. autoproperty:: culebra.trainer.abc.DistributedTrainer._default_representation_selection_func_params
.. autoproperty:: culebra.trainer.abc.DistributedTrainer._default_representation_size
.. autoproperty:: culebra.trainer.abc.DistributedTrainer._default_representation_topology_func
.. autoproperty:: culebra.trainer.abc.DistributedTrainer._default_representation_topology_func_params
.. autoproperty:: culebra.trainer.abc.DistributedTrainer._default_verbosity
.. autoproperty:: culebra.trainer.abc.DistributedTrainer._subtrainer_suffixes

Static methods
--------------
.. automethod:: culebra.trainer.abc.DistributedTrainer.receive_representatives
.. automethod:: culebra.trainer.abc.DistributedTrainer.send_representatives

Methods
-------
.. automethod:: culebra.trainer.abc.DistributedTrainer.best_representatives
.. automethod:: culebra.trainer.abc.DistributedTrainer.best_solutions
.. automethod:: culebra.trainer.abc.DistributedTrainer.dump
.. automethod:: culebra.trainer.abc.DistributedTrainer.evaluate
.. automethod:: culebra.trainer.abc.DistributedTrainer.reset
.. automethod:: culebra.trainer.abc.DistributedTrainer.test
.. automethod:: culebra.trainer.abc.DistributedTrainer.train

Private methods
---------------
.. automethod:: culebra.trainer.abc.DistributedTrainer._default_termination_func
.. automethod:: culebra.trainer.abc.DistributedTrainer._do_iteration
.. automethod:: culebra.trainer.abc.DistributedTrainer._do_iteration_stats
.. automethod:: culebra.trainer.abc.DistributedTrainer._finish_iteration
.. automethod:: culebra.trainer.abc.DistributedTrainer._finish_search
.. automethod:: culebra.trainer.abc.DistributedTrainer._generate_subtrainers
.. automethod:: culebra.trainer.abc.DistributedTrainer._get_state
.. automethod:: culebra.trainer.abc.DistributedTrainer._init_internals
.. automethod:: culebra.trainer.abc.DistributedTrainer._init_representatives
.. automethod:: culebra.trainer.abc.DistributedTrainer._init_search
.. automethod:: culebra.trainer.abc.DistributedTrainer._init_state
.. automethod:: culebra.trainer.abc.DistributedTrainer._load_state
.. automethod:: culebra.trainer.abc.DistributedTrainer._new_state
.. automethod:: culebra.trainer.abc.DistributedTrainer._postprocess_iteration
.. automethod:: culebra.trainer.abc.DistributedTrainer._preprocess_iteration
.. automethod:: culebra.trainer.abc.DistributedTrainer._reset_internals
.. automethod:: culebra.trainer.abc.DistributedTrainer._reset_state
.. automethod:: culebra.trainer.abc.DistributedTrainer._save_state
.. automethod:: culebra.trainer.abc.DistributedTrainer._search
.. automethod:: culebra.trainer.abc.DistributedTrainer._set_cooperative_fitness
.. automethod:: culebra.trainer.abc.DistributedTrainer._set_state
.. automethod:: culebra.trainer.abc.DistributedTrainer._start_iteration
.. automethod:: culebra.trainer.abc.DistributedTrainer._termination_criterion
