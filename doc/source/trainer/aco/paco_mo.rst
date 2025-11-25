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

:class:`culebra.trainer.aco.PACOMO` class
=========================================

.. autoclass:: culebra.trainer.aco.PACOMO

Class attributes
----------------
.. autoattribute:: culebra.trainer.aco.PACOMO.objective_stats
.. autoattribute:: culebra.trainer.aco.PACOMO.stats_names

Class methods
-------------
.. automethod:: culebra.trainer.aco.PACOMO.load

Properties
----------
.. autoproperty:: culebra.trainer.aco.PACOMO.checkpoint_enable
.. autoproperty:: culebra.trainer.aco.PACOMO.checkpoint_filename
.. autoproperty:: culebra.trainer.aco.PACOMO.checkpoint_freq
.. autoproperty:: culebra.trainer.aco.PACOMO.choice_info
.. autoproperty:: culebra.trainer.aco.PACOMO.col
.. autoproperty:: culebra.trainer.aco.PACOMO.col_size
.. autoproperty:: culebra.trainer.aco.PACOMO.container
.. autoproperty:: culebra.trainer.aco.PACOMO.current_iter
.. autoproperty:: culebra.trainer.aco.PACOMO.custom_termination_func
.. autoproperty:: culebra.trainer.aco.PACOMO.exploitation_prob
.. autoproperty:: culebra.trainer.aco.PACOMO.fitness_function
.. autoproperty:: culebra.trainer.aco.PACOMO.heuristic
.. autoproperty:: culebra.trainer.aco.PACOMO.heuristic_influence
.. autoproperty:: culebra.trainer.aco.PACOMO.heuristic_shapes
.. autoproperty:: culebra.trainer.aco.PACOMO.index
.. autoproperty:: culebra.trainer.aco.PACOMO.initial_pheromone
.. autoproperty:: culebra.trainer.aco.PACOMO.logbook
.. autoproperty:: culebra.trainer.aco.PACOMO.max_num_iters
.. autoproperty:: culebra.trainer.aco.PACOMO.max_pheromone
.. autoproperty:: culebra.trainer.aco.PACOMO.num_evals
.. autoproperty:: culebra.trainer.aco.PACOMO.num_heuristic_matrices
.. autoproperty:: culebra.trainer.aco.PACOMO.num_pheromone_matrices
.. autoproperty:: culebra.trainer.aco.PACOMO.pheromone
.. autoproperty:: culebra.trainer.aco.PACOMO.pheromone_influence
.. autoproperty:: culebra.trainer.aco.PACOMO.pheromone_shapes
.. autoproperty:: culebra.trainer.aco.PACOMO.pop
.. autoproperty:: culebra.trainer.aco.PACOMO.pop_size
.. autoproperty:: culebra.trainer.aco.PACOMO.random_seed
.. autoproperty:: culebra.trainer.aco.PACOMO.representatives
.. autoproperty:: culebra.trainer.aco.PACOMO.runtime
.. autoproperty:: culebra.trainer.aco.PACOMO.solution_cls
.. autoproperty:: culebra.trainer.aco.PACOMO.species
.. autoproperty:: culebra.trainer.aco.PACOMO.verbose

Methods
-------
.. automethod:: culebra.trainer.aco.PACOMO.best_representatives
.. automethod:: culebra.trainer.aco.PACOMO.best_solutions
.. automethod:: culebra.trainer.aco.PACOMO.dump
.. automethod:: culebra.trainer.aco.PACOMO.evaluate
.. automethod:: culebra.trainer.aco.PACOMO.reset
.. automethod:: culebra.trainer.aco.PACOMO.test
.. automethod:: culebra.trainer.aco.PACOMO.train

Private methods
---------------
.. automethod:: culebra.trainer.aco.PACOMO._ant_choice_info
.. automethod:: culebra.trainer.aco.PACOMO._calculate_choice_info
.. automethod:: culebra.trainer.aco.PACOMO._decrease_pheromone
.. automethod:: culebra.trainer.aco.PACOMO._default_termination_func
.. automethod:: culebra.trainer.aco.PACOMO._deposit_pheromone
.. automethod:: culebra.trainer.aco.PACOMO._do_iteration
.. automethod:: culebra.trainer.aco.PACOMO._do_iteration_stats
.. automethod:: culebra.trainer.aco.PACOMO._finish_iteration
.. automethod:: culebra.trainer.aco.PACOMO._finish_search
.. automethod:: culebra.trainer.aco.PACOMO._generate_ant
.. automethod:: culebra.trainer.aco.PACOMO._generate_col
.. automethod:: culebra.trainer.aco.PACOMO._get_state
.. automethod:: culebra.trainer.aco.PACOMO._increase_pheromone
.. automethod:: culebra.trainer.aco.PACOMO._init_internals
.. automethod:: culebra.trainer.aco.PACOMO._init_pheromone
.. automethod:: culebra.trainer.aco.PACOMO._init_representatives
.. automethod:: culebra.trainer.aco.PACOMO._init_search
.. automethod:: culebra.trainer.aco.PACOMO._init_state
.. automethod:: culebra.trainer.aco.PACOMO._load_state
.. automethod:: culebra.trainer.aco.PACOMO._new_state
.. automethod:: culebra.trainer.aco.PACOMO._next_choice
.. automethod:: culebra.trainer.aco.PACOMO._pheromone_amount
.. automethod:: culebra.trainer.aco.PACOMO._postprocess_iteration
.. automethod:: culebra.trainer.aco.PACOMO._preprocess_iteration
.. automethod:: culebra.trainer.aco.PACOMO._reset_internals
.. automethod:: culebra.trainer.aco.PACOMO._reset_state
.. automethod:: culebra.trainer.aco.PACOMO._save_state
.. automethod:: culebra.trainer.aco.PACOMO._search
.. automethod:: culebra.trainer.aco.PACOMO._set_cooperative_fitness
.. automethod:: culebra.trainer.aco.PACOMO._set_state
.. automethod:: culebra.trainer.aco.PACOMO._start_iteration
.. automethod:: culebra.trainer.aco.PACOMO._termination_criterion
.. automethod:: culebra.trainer.aco.PACOMO._update_elite
.. automethod:: culebra.trainer.aco.PACOMO._update_pheromone
.. automethod:: culebra.trainer.aco.PACOMO._update_pop
