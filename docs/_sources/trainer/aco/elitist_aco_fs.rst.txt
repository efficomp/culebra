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

:class:`culebra.trainer.aco.ElitistACOFS` class
===============================================

.. autoclass:: culebra.trainer.aco.ElitistACOFS

Class attributes
----------------
.. autoattribute:: culebra.trainer.aco.ElitistACOFS.objective_stats
.. autoattribute:: culebra.trainer.aco.ElitistACOFS.stats_names

Class methods
-------------
.. automethod:: culebra.trainer.aco.ElitistACOFS.load

Properties
----------
.. autoproperty:: culebra.trainer.aco.ElitistACOFS.checkpoint_activation
.. autoproperty:: culebra.trainer.aco.ElitistACOFS.checkpoint_filename
.. autoproperty:: culebra.trainer.aco.ElitistACOFS.checkpoint_freq
.. autoproperty:: culebra.trainer.aco.ElitistACOFS.choice_info
.. autoproperty:: culebra.trainer.aco.ElitistACOFS.col
.. autoproperty:: culebra.trainer.aco.ElitistACOFS.col_size
.. autoproperty:: culebra.trainer.aco.ElitistACOFS.container
.. autoproperty:: culebra.trainer.aco.ElitistACOFS.current_iter
.. autoproperty:: culebra.trainer.aco.ElitistACOFS.custom_termination_func
.. autoproperty:: culebra.trainer.aco.ElitistACOFS.discard_prob
.. autoproperty:: culebra.trainer.aco.ElitistACOFS.exploitation_prob
.. autoproperty:: culebra.trainer.aco.ElitistACOFS.fitness_function
.. autoproperty:: culebra.trainer.aco.ElitistACOFS.heuristic
.. autoproperty:: culebra.trainer.aco.ElitistACOFS.heuristic_influence
.. autoproperty:: culebra.trainer.aco.ElitistACOFS.heuristic_shapes
.. autoproperty:: culebra.trainer.aco.ElitistACOFS.index
.. autoproperty:: culebra.trainer.aco.ElitistACOFS.initial_pheromone
.. autoproperty:: culebra.trainer.aco.ElitistACOFS.logbook
.. autoproperty:: culebra.trainer.aco.ElitistACOFS.max_num_iters
.. autoproperty:: culebra.trainer.aco.ElitistACOFS.num_evals
.. autoproperty:: culebra.trainer.aco.ElitistACOFS.num_heuristic_matrices
.. autoproperty:: culebra.trainer.aco.ElitistACOFS.num_pheromone_matrices
.. autoproperty:: culebra.trainer.aco.ElitistACOFS.pheromone
.. autoproperty:: culebra.trainer.aco.ElitistACOFS.pheromone_influence
.. autoproperty:: culebra.trainer.aco.ElitistACOFS.pheromone_shapes
.. autoproperty:: culebra.trainer.aco.ElitistACOFS.random_seed
.. autoproperty:: culebra.trainer.aco.ElitistACOFS.representatives
.. autoproperty:: culebra.trainer.aco.ElitistACOFS.runtime
.. autoproperty:: culebra.trainer.aco.ElitistACOFS.solution_cls
.. autoproperty:: culebra.trainer.aco.ElitistACOFS.species
.. autoproperty:: culebra.trainer.aco.ElitistACOFS.verbosity

Private properties
------------------
.. autoproperty:: culebra.trainer.aco.ElitistACOFS._default_checkpoint_activation
.. autoproperty:: culebra.trainer.aco.ElitistACOFS._default_checkpoint_filename
.. autoproperty:: culebra.trainer.aco.ElitistACOFS._default_checkpoint_freq
.. autoproperty:: culebra.trainer.aco.ElitistACOFS._default_col_size
.. autoproperty:: culebra.trainer.aco.ElitistACOFS._default_discard_prob
.. autoproperty:: culebra.trainer.aco.ElitistACOFS._default_exploitation_prob
.. autoproperty:: culebra.trainer.aco.ElitistACOFS._default_heuristic
.. autoproperty:: culebra.trainer.aco.ElitistACOFS._default_heuristic_influence
.. autoproperty:: culebra.trainer.aco.ElitistACOFS._default_index
.. autoproperty:: culebra.trainer.aco.ElitistACOFS._default_initial_pheromone
.. autoproperty:: culebra.trainer.aco.ElitistACOFS._default_max_num_iters
.. autoproperty:: culebra.trainer.aco.ElitistACOFS._default_pheromone_influence
.. autoproperty:: culebra.trainer.aco.ElitistACOFS._default_verbosity

Methods
-------
.. automethod:: culebra.trainer.aco.ElitistACOFS.train
.. automethod:: culebra.trainer.aco.ElitistACOFS.test
.. automethod:: culebra.trainer.aco.ElitistACOFS.reset
.. automethod:: culebra.trainer.aco.ElitistACOFS.evaluate
.. automethod:: culebra.trainer.aco.ElitistACOFS.dump
.. automethod:: culebra.trainer.aco.ElitistACOFS.best_solutions
.. automethod:: culebra.trainer.aco.ElitistACOFS.best_representatives

Private methods
---------------
.. automethod:: culebra.trainer.aco.ElitistACOFS._ant_choice_info
.. automethod:: culebra.trainer.aco.ElitistACOFS._calculate_choice_info
.. automethod:: culebra.trainer.aco.ElitistACOFS._default_termination_func
.. automethod:: culebra.trainer.aco.ElitistACOFS._deposit_pheromone
.. automethod:: culebra.trainer.aco.ElitistACOFS._do_iteration
.. automethod:: culebra.trainer.aco.ElitistACOFS._do_iteration_stats
.. automethod:: culebra.trainer.aco.ElitistACOFS._finish_iteration
.. automethod:: culebra.trainer.aco.ElitistACOFS._finish_search
.. automethod:: culebra.trainer.aco.ElitistACOFS._generate_ant
.. automethod:: culebra.trainer.aco.ElitistACOFS._generate_col
.. automethod:: culebra.trainer.aco.ElitistACOFS._get_state
.. automethod:: culebra.trainer.aco.ElitistACOFS._init_internals
.. automethod:: culebra.trainer.aco.ElitistACOFS._init_pheromone
.. automethod:: culebra.trainer.aco.ElitistACOFS._init_representatives
.. automethod:: culebra.trainer.aco.ElitistACOFS._init_search
.. automethod:: culebra.trainer.aco.ElitistACOFS._init_state
.. automethod:: culebra.trainer.aco.ElitistACOFS._load_state
.. automethod:: culebra.trainer.aco.ElitistACOFS._new_state
.. automethod:: culebra.trainer.aco.ElitistACOFS._next_choice
.. automethod:: culebra.trainer.aco.ElitistACOFS._pheromone_amount
.. automethod:: culebra.trainer.aco.ElitistACOFS._postprocess_iteration
.. automethod:: culebra.trainer.aco.ElitistACOFS._preprocess_iteration
.. automethod:: culebra.trainer.aco.ElitistACOFS._reset_internals
.. automethod:: culebra.trainer.aco.ElitistACOFS._reset_state
.. automethod:: culebra.trainer.aco.ElitistACOFS._save_state
.. automethod:: culebra.trainer.aco.ElitistACOFS._search
.. automethod:: culebra.trainer.aco.ElitistACOFS._set_cooperative_fitness
.. automethod:: culebra.trainer.aco.ElitistACOFS._set_state
.. automethod:: culebra.trainer.aco.ElitistACOFS._start_iteration
.. automethod:: culebra.trainer.aco.ElitistACOFS._termination_criterion
.. automethod:: culebra.trainer.aco.ElitistACOFS._update_elite
.. automethod:: culebra.trainer.aco.ElitistACOFS._update_pheromone
