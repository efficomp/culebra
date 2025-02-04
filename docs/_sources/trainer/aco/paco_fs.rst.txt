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

:py:class:`culebra.trainer.aco.PACO_FS` class
=============================================

.. autoclass:: culebra.trainer.aco.PACO_FS

Class attributes
----------------
.. autoattribute:: culebra.trainer.aco.PACO_FS.stats_names
.. autoattribute:: culebra.trainer.aco.PACO_FS.objective_stats

Properties
----------
.. autoproperty:: culebra.trainer.aco.PACO_FS.num_pheromone_matrices
.. autoproperty:: culebra.trainer.aco.PACO_FS.num_heuristic_matrices
.. autoproperty:: culebra.trainer.aco.PACO_FS.solution_cls
.. autoproperty:: culebra.trainer.aco.PACO_FS.species
.. autoproperty:: culebra.trainer.aco.PACO_FS.fitness_function
.. autoproperty:: culebra.trainer.aco.PACO_FS.initial_pheromone
.. autoproperty:: culebra.trainer.aco.PACO_FS.heuristic
.. autoproperty:: culebra.trainer.aco.PACO_FS.pheromone_influence
.. autoproperty:: culebra.trainer.aco.PACO_FS.heuristic_influence
.. autoproperty:: culebra.trainer.aco.PACO_FS.pheromone
.. autoproperty:: culebra.trainer.aco.PACO_FS.choice_info
.. autoproperty:: culebra.trainer.aco.PACO_FS.max_num_iters
.. autoproperty:: culebra.trainer.aco.PACO_FS.current_iter
.. autoproperty:: culebra.trainer.aco.PACO_FS.custom_termination_func
.. autoproperty:: culebra.trainer.aco.PACO_FS.col_size
.. autoproperty:: culebra.trainer.aco.PACO_FS.pop_size
.. autoproperty:: culebra.trainer.aco.PACO_FS.checkpoint_enable
.. autoproperty:: culebra.trainer.aco.PACO_FS.checkpoint_freq
.. autoproperty:: culebra.trainer.aco.PACO_FS.checkpoint_filename
.. autoproperty:: culebra.trainer.aco.PACO_FS.verbose
.. autoproperty:: culebra.trainer.aco.PACO_FS.random_seed
.. autoproperty:: culebra.trainer.aco.PACO_FS.logbook
.. autoproperty:: culebra.trainer.aco.PACO_FS.num_evals
.. autoproperty:: culebra.trainer.aco.PACO_FS.runtime
.. autoproperty:: culebra.trainer.aco.PACO_FS.index
.. autoproperty:: culebra.trainer.aco.PACO_FS.container
.. autoproperty:: culebra.trainer.aco.PACO_FS.representatives
.. autoproperty:: culebra.trainer.aco.PACO_FS.col
.. autoproperty:: culebra.trainer.aco.PACO_FS.pop
.. autoproperty:: culebra.trainer.aco.PACO_FS.discard_prob

Methods
-------
.. automethod:: culebra.trainer.aco.PACO_FS.reset
.. automethod:: culebra.trainer.aco.PACO_FS.evaluate
.. automethod:: culebra.trainer.aco.PACO_FS.best_solutions
.. automethod:: culebra.trainer.aco.PACO_FS.best_representatives
.. automethod:: culebra.trainer.aco.PACO_FS.train
.. automethod:: culebra.trainer.aco.PACO_FS.test

Private methods
---------------
.. automethod:: culebra.trainer.aco.PACO_FS._get_state
.. automethod:: culebra.trainer.aco.PACO_FS._set_state
.. automethod:: culebra.trainer.aco.PACO_FS._save_state
.. automethod:: culebra.trainer.aco.PACO_FS._load_state
.. automethod:: culebra.trainer.aco.PACO_FS._new_state
.. automethod:: culebra.trainer.aco.PACO_FS._init_state
.. automethod:: culebra.trainer.aco.PACO_FS._reset_state
.. automethod:: culebra.trainer.aco.PACO_FS._init_internals
.. automethod:: culebra.trainer.aco.PACO_FS._reset_internals
.. automethod:: culebra.trainer.aco.PACO_FS._init_search
.. automethod:: culebra.trainer.aco.PACO_FS._search
.. automethod:: culebra.trainer.aco.PACO_FS._finish_search
.. automethod:: culebra.trainer.aco.PACO_FS._start_iteration
.. automethod:: culebra.trainer.aco.PACO_FS._preprocess_iteration
.. automethod:: culebra.trainer.aco.PACO_FS._do_iteration
.. automethod:: culebra.trainer.aco.PACO_FS._postprocess_iteration
.. automethod:: culebra.trainer.aco.PACO_FS._finish_iteration
.. automethod:: culebra.trainer.aco.PACO_FS._do_iteration_stats
.. automethod:: culebra.trainer.aco.PACO_FS._default_termination_func
.. automethod:: culebra.trainer.aco.PACO_FS._termination_criterion
.. automethod:: culebra.trainer.aco.PACO_FS._init_representatives
.. automethod:: culebra.trainer.aco.PACO_FS._calculate_choice_info
.. automethod:: culebra.trainer.aco.PACO_FS._initial_choice
.. automethod:: culebra.trainer.aco.PACO_FS._next_choice
.. automethod:: culebra.trainer.aco.PACO_FS._feasible_neighborhood_probs
.. automethod:: culebra.trainer.aco.PACO_FS._generate_ant
.. automethod:: culebra.trainer.aco.PACO_FS._generate_col
.. automethod:: culebra.trainer.aco.PACO_FS._init_pheromone
.. automethod:: culebra.trainer.aco.PACO_FS._deposit_pheromone
.. automethod:: culebra.trainer.aco.PACO_FS._increase_pheromone
.. automethod:: culebra.trainer.aco.PACO_FS._decrease_pheromone
.. automethod:: culebra.trainer.aco.PACO_FS._update_pheromone
.. automethod:: culebra.trainer.aco.PACO_FS._update_pop

