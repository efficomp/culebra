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

:class:`culebra.trainer.abc.CooperativeTrainer` class
=====================================================

.. autoclass:: culebra.trainer.abc.CooperativeTrainer

Class attributes
----------------
.. autoattribute:: culebra.trainer.abc.CooperativeTrainer.objective_stats
.. autoattribute:: culebra.trainer.abc.CooperativeTrainer.stats_names

Class methods
-------------
.. automethod:: culebra.trainer.abc.CooperativeTrainer.load

Properties
----------
.. autoproperty:: culebra.trainer.abc.CooperativeTrainer.checkpoint_activation
.. autoproperty:: culebra.trainer.abc.CooperativeTrainer.checkpoint_filename
.. autoproperty:: culebra.trainer.abc.CooperativeTrainer.checkpoint_freq
.. autoproperty:: culebra.trainer.abc.CooperativeTrainer.container
.. autoproperty:: culebra.trainer.abc.CooperativeTrainer.current_iter
.. autoproperty:: culebra.trainer.abc.CooperativeTrainer.custom_termination_func
.. autoproperty:: culebra.trainer.abc.CooperativeTrainer.fitness_function
.. autoproperty:: culebra.trainer.abc.CooperativeTrainer.index
.. autoproperty:: culebra.trainer.abc.CooperativeTrainer.logbook
.. autoproperty:: culebra.trainer.abc.CooperativeTrainer.max_num_iters
.. autoproperty:: culebra.trainer.abc.CooperativeTrainer.num_evals
.. autoproperty:: culebra.trainer.abc.CooperativeTrainer.num_subtrainers
.. autoproperty:: culebra.trainer.abc.CooperativeTrainer.random_seed
.. autoproperty:: culebra.trainer.abc.CooperativeTrainer.representation_freq
.. autoproperty:: culebra.trainer.abc.CooperativeTrainer.representation_selection_func
.. autoproperty:: culebra.trainer.abc.CooperativeTrainer.representation_selection_func_params
.. autoproperty:: culebra.trainer.abc.CooperativeTrainer.representation_size
.. autoproperty:: culebra.trainer.abc.CooperativeTrainer.representation_topology_func
.. autoproperty:: culebra.trainer.abc.CooperativeTrainer.representation_topology_func_params
.. autoproperty:: culebra.trainer.abc.CooperativeTrainer.representatives
.. autoproperty:: culebra.trainer.abc.CooperativeTrainer.runtime
.. autoproperty:: culebra.trainer.abc.CooperativeTrainer.solution_classes
.. autoproperty:: culebra.trainer.abc.CooperativeTrainer.species
.. autoproperty:: culebra.trainer.abc.CooperativeTrainer.subtrainer_checkpoint_filenames
.. autoproperty:: culebra.trainer.abc.CooperativeTrainer.subtrainer_cls
.. autoproperty:: culebra.trainer.abc.CooperativeTrainer.subtrainer_params
.. autoproperty:: culebra.trainer.abc.CooperativeTrainer.subtrainers
.. autoproperty:: culebra.trainer.abc.CooperativeTrainer.verbosity

Private properties
------------------
.. autoproperty:: culebra.trainer.abc.CooperativeTrainer._default_checkpoint_activation
.. autoproperty:: culebra.trainer.abc.CooperativeTrainer._default_checkpoint_filename
.. autoproperty:: culebra.trainer.abc.CooperativeTrainer._default_checkpoint_freq
.. autoproperty:: culebra.trainer.abc.CooperativeTrainer._default_index
.. autoproperty:: culebra.trainer.abc.CooperativeTrainer._default_max_num_iters
.. autoproperty:: culebra.trainer.abc.CooperativeTrainer._default_num_subtrainers
.. autoproperty:: culebra.trainer.abc.CooperativeTrainer._default_representation_freq
.. autoproperty:: culebra.trainer.abc.CooperativeTrainer._default_representation_selection_func
.. autoproperty:: culebra.trainer.abc.CooperativeTrainer._default_representation_selection_func_params
.. autoproperty:: culebra.trainer.abc.CooperativeTrainer._default_representation_size
.. autoproperty:: culebra.trainer.abc.CooperativeTrainer._default_representation_topology_func
.. autoproperty:: culebra.trainer.abc.CooperativeTrainer._default_representation_topology_func_params
.. autoproperty:: culebra.trainer.abc.CooperativeTrainer._default_verbosity
.. autoproperty:: culebra.trainer.abc.CooperativeTrainer._subtrainer_suffixes

Static methods
--------------
.. automethod:: culebra.trainer.abc.CooperativeTrainer.receive_representatives
.. automethod:: culebra.trainer.abc.CooperativeTrainer.send_representatives

Private static methods
----------------------
.. automethod:: culebra.trainer.abc.CooperativeTrainer._init_subtrainer_representatives

Methods
-------
.. automethod:: culebra.trainer.abc.CooperativeTrainer.best_representatives
.. automethod:: culebra.trainer.abc.CooperativeTrainer.best_solutions
.. automethod:: culebra.trainer.abc.CooperativeTrainer.dump
.. automethod:: culebra.trainer.abc.CooperativeTrainer.evaluate
.. automethod:: culebra.trainer.abc.CooperativeTrainer.reset
.. automethod:: culebra.trainer.abc.CooperativeTrainer.test
.. automethod:: culebra.trainer.abc.CooperativeTrainer.train

Private methods
---------------
.. automethod:: culebra.trainer.abc.CooperativeTrainer._default_termination_func
.. automethod:: culebra.trainer.abc.CooperativeTrainer._do_iteration
.. automethod:: culebra.trainer.abc.CooperativeTrainer._do_iteration_stats
.. automethod:: culebra.trainer.abc.CooperativeTrainer._finish_iteration
.. automethod:: culebra.trainer.abc.CooperativeTrainer._finish_search
.. automethod:: culebra.trainer.abc.CooperativeTrainer._generate_subtrainers
.. automethod:: culebra.trainer.abc.CooperativeTrainer._get_state
.. automethod:: culebra.trainer.abc.CooperativeTrainer._init_internals
.. automethod:: culebra.trainer.abc.CooperativeTrainer._init_representatives
.. automethod:: culebra.trainer.abc.CooperativeTrainer._init_search
.. automethod:: culebra.trainer.abc.CooperativeTrainer._init_state
.. automethod:: culebra.trainer.abc.CooperativeTrainer._load_state
.. automethod:: culebra.trainer.abc.CooperativeTrainer._new_state
.. automethod:: culebra.trainer.abc.CooperativeTrainer._postprocess_iteration
.. automethod:: culebra.trainer.abc.CooperativeTrainer._preprocess_iteration
.. automethod:: culebra.trainer.abc.CooperativeTrainer._reset_internals
.. automethod:: culebra.trainer.abc.CooperativeTrainer._reset_state
.. automethod:: culebra.trainer.abc.CooperativeTrainer._save_state
.. automethod:: culebra.trainer.abc.CooperativeTrainer._search
.. automethod:: culebra.trainer.abc.CooperativeTrainer._set_cooperative_fitness
.. automethod:: culebra.trainer.abc.CooperativeTrainer._set_state
.. automethod:: culebra.trainer.abc.CooperativeTrainer._start_iteration
.. automethod:: culebra.trainer.abc.CooperativeTrainer._termination_criterion
