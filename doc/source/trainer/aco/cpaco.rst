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

:py:class:`culebra.trainer.aco.CPACO` class
===========================================

.. autoclass:: culebra.trainer.aco.CPACO

Class attributes
----------------
.. autoattribute:: culebra.trainer.aco.CPACO.stats_names
.. autoattribute:: culebra.trainer.aco.CPACO.objective_stats

Class methods
-------------
.. automethod:: culebra.trainer.aco.CPACO.load_pickle

Properties
----------
.. autoproperty:: culebra.trainer.aco.CPACO.num_pheromone_matrices
.. autoproperty:: culebra.trainer.aco.CPACO.num_heuristic_matrices
.. autoproperty:: culebra.trainer.aco.CPACO.solution_cls
.. autoproperty:: culebra.trainer.aco.CPACO.species
.. autoproperty:: culebra.trainer.aco.CPACO.fitness_function
.. autoproperty:: culebra.trainer.aco.CPACO.initial_pheromone
.. autoproperty:: culebra.trainer.aco.CPACO.heuristic
.. autoproperty:: culebra.trainer.aco.CPACO.pheromone_influence
.. autoproperty:: culebra.trainer.aco.CPACO.heuristic_influence
.. autoproperty:: culebra.trainer.aco.CPACO.pheromone
.. autoproperty:: culebra.trainer.aco.CPACO.choice_info
.. autoproperty:: culebra.trainer.aco.CPACO.max_num_iters
.. autoproperty:: culebra.trainer.aco.CPACO.current_iter
.. autoproperty:: culebra.trainer.aco.CPACO.custom_termination_func
.. autoproperty:: culebra.trainer.aco.CPACO.col_size
.. autoproperty:: culebra.trainer.aco.CPACO.pop_size
.. autoproperty:: culebra.trainer.aco.CPACO.checkpoint_enable
.. autoproperty:: culebra.trainer.aco.CPACO.checkpoint_freq
.. autoproperty:: culebra.trainer.aco.CPACO.checkpoint_filename
.. autoproperty:: culebra.trainer.aco.CPACO.verbose
.. autoproperty:: culebra.trainer.aco.CPACO.random_seed
.. autoproperty:: culebra.trainer.aco.CPACO.logbook
.. autoproperty:: culebra.trainer.aco.CPACO.num_evals
.. autoproperty:: culebra.trainer.aco.CPACO.runtime
.. autoproperty:: culebra.trainer.aco.CPACO.index
.. autoproperty:: culebra.trainer.aco.CPACO.container
.. autoproperty:: culebra.trainer.aco.CPACO.representatives
.. autoproperty:: culebra.trainer.aco.CPACO.col
.. autoproperty:: culebra.trainer.aco.CPACO.pop

Methods
-------
.. automethod:: culebra.trainer.aco.CPACO.save_pickle
.. automethod:: culebra.trainer.aco.CPACO.reset
.. automethod:: culebra.trainer.aco.CPACO.evaluate
.. automethod:: culebra.trainer.aco.CPACO.best_solutions
.. automethod:: culebra.trainer.aco.CPACO.best_representatives
.. automethod:: culebra.trainer.aco.CPACO.train
.. automethod:: culebra.trainer.aco.CPACO.test

Private methods
---------------
.. automethod:: culebra.trainer.aco.CPACO._get_state
.. automethod:: culebra.trainer.aco.CPACO._set_state
.. automethod:: culebra.trainer.aco.CPACO._save_state
.. automethod:: culebra.trainer.aco.CPACO._load_state
.. automethod:: culebra.trainer.aco.CPACO._new_state
.. automethod:: culebra.trainer.aco.CPACO._init_state
.. automethod:: culebra.trainer.aco.CPACO._reset_state
.. automethod:: culebra.trainer.aco.CPACO._init_internals
.. automethod:: culebra.trainer.aco.CPACO._reset_internals
.. automethod:: culebra.trainer.aco.CPACO._init_search
.. automethod:: culebra.trainer.aco.CPACO._search
.. automethod:: culebra.trainer.aco.CPACO._finish_search
.. automethod:: culebra.trainer.aco.CPACO._start_iteration
.. automethod:: culebra.trainer.aco.CPACO._preprocess_iteration
.. automethod:: culebra.trainer.aco.CPACO._do_iteration
.. automethod:: culebra.trainer.aco.CPACO._postprocess_iteration
.. automethod:: culebra.trainer.aco.CPACO._finish_iteration
.. automethod:: culebra.trainer.aco.CPACO._do_iteration_stats
.. automethod:: culebra.trainer.aco.CPACO._default_termination_func
.. automethod:: culebra.trainer.aco.CPACO._termination_criterion
.. automethod:: culebra.trainer.aco.CPACO._init_representatives
.. automethod:: culebra.trainer.aco.CPACO._calculate_choice_info
.. automethod:: culebra.trainer.aco.CPACO._initial_choice
.. automethod:: culebra.trainer.aco.CPACO._next_choice
.. automethod:: culebra.trainer.aco.CPACO._feasible_neighborhood_probs
.. automethod:: culebra.trainer.aco.CPACO._generate_ant
.. automethod:: culebra.trainer.aco.CPACO._generate_col
.. automethod:: culebra.trainer.aco.CPACO._init_pheromone
.. automethod:: culebra.trainer.aco.CPACO._deposit_pheromone
.. automethod:: culebra.trainer.aco.CPACO._increase_pheromone
.. automethod:: culebra.trainer.aco.CPACO._decrease_pheromone
.. automethod:: culebra.trainer.aco.CPACO._update_pheromone
.. automethod:: culebra.trainer.aco.CPACO._update_pop

