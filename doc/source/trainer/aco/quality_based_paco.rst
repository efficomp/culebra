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

:py:class:`culebra.trainer.aco.AgeBasedPACO` class
==================================================

.. autoclass:: culebra.trainer.aco.AgeBasedPACO

Class attributes
----------------
.. autoattribute:: culebra.trainer.aco.AgeBasedPACO.stats_names
.. autoattribute:: culebra.trainer.aco.AgeBasedPACO.objective_stats

Class methods
-------------
.. automethod:: culebra.trainer.aco.AgeBasedPACO.load

Properties
----------
.. autoproperty:: culebra.trainer.aco.AgeBasedPACO.num_pheromone_matrices
.. autoproperty:: culebra.trainer.aco.AgeBasedPACO.num_heuristic_matrices
.. autoproperty:: culebra.trainer.aco.AgeBasedPACO.solution_cls
.. autoproperty:: culebra.trainer.aco.AgeBasedPACO.species
.. autoproperty:: culebra.trainer.aco.AgeBasedPACO.fitness_function
.. autoproperty:: culebra.trainer.aco.AgeBasedPACO.initial_pheromone
.. autoproperty:: culebra.trainer.aco.AgeBasedPACO.max_pheromone
.. autoproperty:: culebra.trainer.aco.AgeBasedPACO.heuristic
.. autoproperty:: culebra.trainer.aco.AgeBasedPACO.pheromone_influence
.. autoproperty:: culebra.trainer.aco.AgeBasedPACO.heuristic_influence
.. autoproperty:: culebra.trainer.aco.AgeBasedPACO.pheromone
.. autoproperty:: culebra.trainer.aco.AgeBasedPACO.choice_info
.. autoproperty:: culebra.trainer.aco.AgeBasedPACO.max_num_iters
.. autoproperty:: culebra.trainer.aco.AgeBasedPACO.current_iter
.. autoproperty:: culebra.trainer.aco.AgeBasedPACO.custom_termination_func
.. autoproperty:: culebra.trainer.aco.AgeBasedPACO.col_size
.. autoproperty:: culebra.trainer.aco.AgeBasedPACO.pop_size
.. autoproperty:: culebra.trainer.aco.AgeBasedPACO.checkpoint_enable
.. autoproperty:: culebra.trainer.aco.AgeBasedPACO.checkpoint_freq
.. autoproperty:: culebra.trainer.aco.AgeBasedPACO.checkpoint_filename
.. autoproperty:: culebra.trainer.aco.AgeBasedPACO.verbose
.. autoproperty:: culebra.trainer.aco.AgeBasedPACO.random_seed
.. autoproperty:: culebra.trainer.aco.AgeBasedPACO.logbook
.. autoproperty:: culebra.trainer.aco.AgeBasedPACO.num_evals
.. autoproperty:: culebra.trainer.aco.AgeBasedPACO.runtime
.. autoproperty:: culebra.trainer.aco.AgeBasedPACO.index
.. autoproperty:: culebra.trainer.aco.AgeBasedPACO.container
.. autoproperty:: culebra.trainer.aco.AgeBasedPACO.representatives
.. autoproperty:: culebra.trainer.aco.AgeBasedPACO.col
.. autoproperty:: culebra.trainer.aco.AgeBasedPACO.pop

Methods
-------
.. automethod:: culebra.trainer.aco.AgeBasedPACO.dump
.. automethod:: culebra.trainer.aco.AgeBasedPACO.reset
.. automethod:: culebra.trainer.aco.AgeBasedPACO.evaluate
.. automethod:: culebra.trainer.aco.AgeBasedPACO.best_solutions
.. automethod:: culebra.trainer.aco.AgeBasedPACO.best_representatives
.. automethod:: culebra.trainer.aco.AgeBasedPACO.train
.. automethod:: culebra.trainer.aco.AgeBasedPACO.test

Private methods
---------------
.. automethod:: culebra.trainer.aco.AgeBasedPACO._get_state
.. automethod:: culebra.trainer.aco.AgeBasedPACO._set_state
.. automethod:: culebra.trainer.aco.AgeBasedPACO._save_state
.. automethod:: culebra.trainer.aco.AgeBasedPACO._load_state
.. automethod:: culebra.trainer.aco.AgeBasedPACO._new_state
.. automethod:: culebra.trainer.aco.AgeBasedPACO._init_state
.. automethod:: culebra.trainer.aco.AgeBasedPACO._reset_state
.. automethod:: culebra.trainer.aco.AgeBasedPACO._init_internals
.. automethod:: culebra.trainer.aco.AgeBasedPACO._reset_internals
.. automethod:: culebra.trainer.aco.AgeBasedPACO._init_search
.. automethod:: culebra.trainer.aco.AgeBasedPACO._search
.. automethod:: culebra.trainer.aco.AgeBasedPACO._finish_search
.. automethod:: culebra.trainer.aco.AgeBasedPACO._start_iteration
.. automethod:: culebra.trainer.aco.AgeBasedPACO._preprocess_iteration
.. automethod:: culebra.trainer.aco.AgeBasedPACO._do_iteration
.. automethod:: culebra.trainer.aco.AgeBasedPACO._postprocess_iteration
.. automethod:: culebra.trainer.aco.AgeBasedPACO._finish_iteration
.. automethod:: culebra.trainer.aco.AgeBasedPACO._do_iteration_stats
.. automethod:: culebra.trainer.aco.AgeBasedPACO._default_termination_func
.. automethod:: culebra.trainer.aco.AgeBasedPACO._termination_criterion
.. automethod:: culebra.trainer.aco.AgeBasedPACO._init_representatives
.. automethod:: culebra.trainer.aco.AgeBasedPACO._calculate_choice_info
.. automethod:: culebra.trainer.aco.AgeBasedPACO._initial_choice
.. automethod:: culebra.trainer.aco.AgeBasedPACO._next_choice
.. automethod:: culebra.trainer.aco.AgeBasedPACO._feasible_neighborhood_probs
.. automethod:: culebra.trainer.aco.AgeBasedPACO._generate_ant
.. automethod:: culebra.trainer.aco.AgeBasedPACO._generate_col
.. automethod:: culebra.trainer.aco.AgeBasedPACO._init_pheromone
.. automethod:: culebra.trainer.aco.AgeBasedPACO._deposit_pheromone
.. automethod:: culebra.trainer.aco.AgeBasedPACO._increase_pheromone
.. automethod:: culebra.trainer.aco.AgeBasedPACO._decrease_pheromone
.. automethod:: culebra.trainer.aco.AgeBasedPACO._update_pheromone
.. automethod:: culebra.trainer.aco.AgeBasedPACO._update_pop

