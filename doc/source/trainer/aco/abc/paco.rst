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

:py:class:`culebra.trainer.aco.abc.PACO` class
==============================================

.. autoclass:: culebra.trainer.aco.abc.PACO

Class attributes
----------------
.. autoattribute:: culebra.trainer.aco.abc.PACO.stats_names
.. autoattribute:: culebra.trainer.aco.abc.PACO.objective_stats

Class methods
-------------
.. automethod:: culebra.trainer.aco.abc.PACO.load_pickle

Properties
----------
.. autoproperty:: culebra.trainer.aco.abc.PACO.num_pheromone_matrices
.. autoproperty:: culebra.trainer.aco.abc.PACO.num_heuristic_matrices
.. autoproperty:: culebra.trainer.aco.abc.PACO.solution_cls
.. autoproperty:: culebra.trainer.aco.abc.PACO.species
.. autoproperty:: culebra.trainer.aco.abc.PACO.fitness_function
.. autoproperty:: culebra.trainer.aco.abc.PACO.initial_pheromone
.. autoproperty:: culebra.trainer.aco.abc.PACO.heuristic
.. autoproperty:: culebra.trainer.aco.abc.PACO.pheromone_influence
.. autoproperty:: culebra.trainer.aco.abc.PACO.heuristic_influence
.. autoproperty:: culebra.trainer.aco.abc.PACO.pheromone
.. autoproperty:: culebra.trainer.aco.abc.PACO.choice_info
.. autoproperty:: culebra.trainer.aco.abc.PACO.max_num_iters
.. autoproperty:: culebra.trainer.aco.abc.PACO.current_iter
.. autoproperty:: culebra.trainer.aco.abc.PACO.custom_termination_func
.. autoproperty:: culebra.trainer.aco.abc.PACO.col_size
.. autoproperty:: culebra.trainer.aco.abc.PACO.pop_size
.. autoproperty:: culebra.trainer.aco.abc.PACO.checkpoint_enable
.. autoproperty:: culebra.trainer.aco.abc.PACO.checkpoint_freq
.. autoproperty:: culebra.trainer.aco.abc.PACO.checkpoint_filename
.. autoproperty:: culebra.trainer.aco.abc.PACO.verbose
.. autoproperty:: culebra.trainer.aco.abc.PACO.random_seed
.. autoproperty:: culebra.trainer.aco.abc.PACO.logbook
.. autoproperty:: culebra.trainer.aco.abc.PACO.num_evals
.. autoproperty:: culebra.trainer.aco.abc.PACO.runtime
.. autoproperty:: culebra.trainer.aco.abc.PACO.index
.. autoproperty:: culebra.trainer.aco.abc.PACO.container
.. autoproperty:: culebra.trainer.aco.abc.PACO.representatives
.. autoproperty:: culebra.trainer.aco.abc.PACO.col
.. autoproperty:: culebra.trainer.aco.abc.PACO.pop

Methods
-------
.. automethod:: culebra.trainer.aco.abc.PACO.save_pickle
.. automethod:: culebra.trainer.aco.abc.PACO.reset
.. automethod:: culebra.trainer.aco.abc.PACO.evaluate
.. automethod:: culebra.trainer.aco.abc.PACO.best_solutions
.. automethod:: culebra.trainer.aco.abc.PACO.best_representatives
.. automethod:: culebra.trainer.aco.abc.PACO.train
.. automethod:: culebra.trainer.aco.abc.PACO.test

Private methods
---------------
.. automethod:: culebra.trainer.aco.abc.PACO._get_state
.. automethod:: culebra.trainer.aco.abc.PACO._set_state
.. automethod:: culebra.trainer.aco.abc.PACO._save_state
.. automethod:: culebra.trainer.aco.abc.PACO._load_state
.. automethod:: culebra.trainer.aco.abc.PACO._new_state
.. automethod:: culebra.trainer.aco.abc.PACO._init_state
.. automethod:: culebra.trainer.aco.abc.PACO._reset_state
.. automethod:: culebra.trainer.aco.abc.PACO._init_internals
.. automethod:: culebra.trainer.aco.abc.PACO._reset_internals
.. automethod:: culebra.trainer.aco.abc.PACO._init_search
.. automethod:: culebra.trainer.aco.abc.PACO._search
.. automethod:: culebra.trainer.aco.abc.PACO._finish_search
.. automethod:: culebra.trainer.aco.abc.PACO._start_iteration
.. automethod:: culebra.trainer.aco.abc.PACO._preprocess_iteration
.. automethod:: culebra.trainer.aco.abc.PACO._do_iteration
.. automethod:: culebra.trainer.aco.abc.PACO._postprocess_iteration
.. automethod:: culebra.trainer.aco.abc.PACO._finish_iteration
.. automethod:: culebra.trainer.aco.abc.PACO._do_iteration_stats
.. automethod:: culebra.trainer.aco.abc.PACO._default_termination_func
.. automethod:: culebra.trainer.aco.abc.PACO._termination_criterion
.. automethod:: culebra.trainer.aco.abc.PACO._init_representatives
.. automethod:: culebra.trainer.aco.abc.PACO._calculate_choice_info
.. automethod:: culebra.trainer.aco.abc.PACO._initial_choice
.. automethod:: culebra.trainer.aco.abc.PACO._next_choice
.. automethod:: culebra.trainer.aco.abc.PACO._feasible_neighborhood_probs
.. automethod:: culebra.trainer.aco.abc.PACO._generate_ant
.. automethod:: culebra.trainer.aco.abc.PACO._generate_col
.. automethod:: culebra.trainer.aco.abc.PACO._init_pheromone
.. automethod:: culebra.trainer.aco.abc.PACO._deposit_pheromone
.. automethod:: culebra.trainer.aco.abc.PACO._increase_pheromone
.. automethod:: culebra.trainer.aco.abc.PACO._decrease_pheromone
.. automethod:: culebra.trainer.aco.abc.PACO._update_pheromone
.. automethod:: culebra.trainer.aco.abc.PACO._update_pop

