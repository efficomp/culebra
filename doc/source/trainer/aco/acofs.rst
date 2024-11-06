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

:py:class:`culebra.trainer.aco.ACOFS` class
===========================================

.. autoclass:: culebra.trainer.aco.ACOFS

Class attributes
----------------
.. autoattribute:: culebra.trainer.aco.ACOFS.stats_names
.. autoattribute:: culebra.trainer.aco.ACOFS.objective_stats

Properties
----------
.. autoproperty:: culebra.trainer.aco.ACOFS.num_pheromone_matrices
.. autoproperty:: culebra.trainer.aco.ACOFS.num_heuristic_matrices
.. autoproperty:: culebra.trainer.aco.ACOFS.solution_cls
.. autoproperty:: culebra.trainer.aco.ACOFS.species
.. autoproperty:: culebra.trainer.aco.ACOFS.fitness_function
.. autoproperty:: culebra.trainer.aco.ACOFS.initial_pheromone
.. autoproperty:: culebra.trainer.aco.ACOFS.heuristic
.. autoproperty:: culebra.trainer.aco.ACOFS.pheromone_influence
.. autoproperty:: culebra.trainer.aco.ACOFS.heuristic_influence
.. autoproperty:: culebra.trainer.aco.ACOFS.pheromone
.. autoproperty:: culebra.trainer.aco.ACOFS.choice_info
.. autoproperty:: culebra.trainer.aco.ACOFS.max_num_iters
.. autoproperty:: culebra.trainer.aco.ACOFS.current_iter
.. autoproperty:: culebra.trainer.aco.ACOFS.custom_termination_func
.. autoproperty:: culebra.trainer.aco.ACOFS.col_size
.. autoproperty:: culebra.trainer.aco.ACOFS.pop_size
.. autoproperty:: culebra.trainer.aco.ACOFS.checkpoint_enable
.. autoproperty:: culebra.trainer.aco.ACOFS.checkpoint_freq
.. autoproperty:: culebra.trainer.aco.ACOFS.checkpoint_filename
.. autoproperty:: culebra.trainer.aco.ACOFS.verbose
.. autoproperty:: culebra.trainer.aco.ACOFS.random_seed
.. autoproperty:: culebra.trainer.aco.ACOFS.logbook
.. autoproperty:: culebra.trainer.aco.ACOFS.num_evals
.. autoproperty:: culebra.trainer.aco.ACOFS.runtime
.. autoproperty:: culebra.trainer.aco.ACOFS.index
.. autoproperty:: culebra.trainer.aco.ACOFS.container
.. autoproperty:: culebra.trainer.aco.ACOFS.representatives
.. autoproperty:: culebra.trainer.aco.ACOFS.col
.. autoproperty:: culebra.trainer.aco.ACOFS.pop
.. autoproperty:: culebra.trainer.aco.ACOFS.discard_prob

Methods
-------
.. automethod:: culebra.trainer.aco.ACOFS.reset
.. automethod:: culebra.trainer.aco.ACOFS.evaluate
.. automethod:: culebra.trainer.aco.ACOFS.best_solutions
.. automethod:: culebra.trainer.aco.ACOFS.best_representatives
.. automethod:: culebra.trainer.aco.ACOFS.train
.. automethod:: culebra.trainer.aco.ACOFS.test

Private methods
---------------
.. automethod:: culebra.trainer.aco.ACOFS._get_state
.. automethod:: culebra.trainer.aco.ACOFS._set_state
.. automethod:: culebra.trainer.aco.ACOFS._save_state
.. automethod:: culebra.trainer.aco.ACOFS._load_state
.. automethod:: culebra.trainer.aco.ACOFS._new_state
.. automethod:: culebra.trainer.aco.ACOFS._init_state
.. automethod:: culebra.trainer.aco.ACOFS._reset_state
.. automethod:: culebra.trainer.aco.ACOFS._init_internals
.. automethod:: culebra.trainer.aco.ACOFS._reset_internals
.. automethod:: culebra.trainer.aco.ACOFS._init_search
.. automethod:: culebra.trainer.aco.ACOFS._search
.. automethod:: culebra.trainer.aco.ACOFS._finish_search
.. automethod:: culebra.trainer.aco.ACOFS._start_iteration
.. automethod:: culebra.trainer.aco.ACOFS._preprocess_iteration
.. automethod:: culebra.trainer.aco.ACOFS._do_iteration
.. automethod:: culebra.trainer.aco.ACOFS._postprocess_iteration
.. automethod:: culebra.trainer.aco.ACOFS._finish_iteration
.. automethod:: culebra.trainer.aco.ACOFS._do_iteration_stats
.. automethod:: culebra.trainer.aco.ACOFS._default_termination_func
.. automethod:: culebra.trainer.aco.ACOFS._termination_criterion
.. automethod:: culebra.trainer.aco.ACOFS._init_representatives
.. automethod:: culebra.trainer.aco.ACOFS._calculate_choice_info
.. automethod:: culebra.trainer.aco.ACOFS._initial_choice
.. automethod:: culebra.trainer.aco.ACOFS._next_choice
.. automethod:: culebra.trainer.aco.ACOFS._feasible_neighborhood_probs
.. automethod:: culebra.trainer.aco.ACOFS._generate_ant
.. automethod:: culebra.trainer.aco.ACOFS._generate_col
.. automethod:: culebra.trainer.aco.ACOFS._init_pheromone
.. automethod:: culebra.trainer.aco.ACOFS._deposit_pheromone
.. automethod:: culebra.trainer.aco.ACOFS._increase_pheromone
.. automethod:: culebra.trainer.aco.ACOFS._decrease_pheromone
.. automethod:: culebra.trainer.aco.ACOFS._update_pheromone
.. automethod:: culebra.trainer.aco.ACOFS._update_pop

