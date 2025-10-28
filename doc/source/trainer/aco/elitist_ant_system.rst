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

:py:class:`culebra.trainer.aco.ElitistAntSystem` class
======================================================

.. autoclass:: culebra.trainer.aco.ElitistAntSystem

Class attributes
----------------
.. autoattribute:: culebra.trainer.aco.ElitistAntSystem.stats_names
.. autoattribute:: culebra.trainer.aco.ElitistAntSystem.objective_stats

Class methods
-------------
.. automethod:: culebra.trainer.aco.ElitistAntSystem.load

Properties
----------
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.num_pheromone_matrices
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.num_heuristic_matrices
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.solution_cls
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.species
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.fitness_function
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.initial_pheromone
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.heuristic
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.pheromone_influence
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.heuristic_influence
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.exploitation_prob
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.pheromone_evaporation_rate
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.elite_weight
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.pheromone
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.choice_info
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.max_num_iters
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.current_iter
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.custom_termination_func
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.col_size
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.checkpoint_enable
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.checkpoint_freq
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.checkpoint_filename
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.verbose
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.random_seed
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.logbook
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.num_evals
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.runtime
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.index
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.container
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.representatives
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.col

Methods
-------
.. automethod:: culebra.trainer.aco.ElitistAntSystem.dump
.. automethod:: culebra.trainer.aco.ElitistAntSystem.reset
.. automethod:: culebra.trainer.aco.ElitistAntSystem.evaluate
.. automethod:: culebra.trainer.aco.ElitistAntSystem.best_solutions
.. automethod:: culebra.trainer.aco.ElitistAntSystem.best_representatives
.. automethod:: culebra.trainer.aco.ElitistAntSystem.train
.. automethod:: culebra.trainer.aco.ElitistAntSystem.test

Private methods
---------------
.. automethod:: culebra.trainer.aco.ElitistAntSystem._get_state
.. automethod:: culebra.trainer.aco.ElitistAntSystem._set_state
.. automethod:: culebra.trainer.aco.ElitistAntSystem._save_state
.. automethod:: culebra.trainer.aco.ElitistAntSystem._load_state
.. automethod:: culebra.trainer.aco.ElitistAntSystem._new_state
.. automethod:: culebra.trainer.aco.ElitistAntSystem._init_state
.. automethod:: culebra.trainer.aco.ElitistAntSystem._reset_state
.. automethod:: culebra.trainer.aco.ElitistAntSystem._init_internals
.. automethod:: culebra.trainer.aco.ElitistAntSystem._reset_internals
.. automethod:: culebra.trainer.aco.ElitistAntSystem._init_search
.. automethod:: culebra.trainer.aco.ElitistAntSystem._search
.. automethod:: culebra.trainer.aco.ElitistAntSystem._finish_search
.. automethod:: culebra.trainer.aco.ElitistAntSystem._start_iteration
.. automethod:: culebra.trainer.aco.ElitistAntSystem._preprocess_iteration
.. automethod:: culebra.trainer.aco.ElitistAntSystem._do_iteration
.. automethod:: culebra.trainer.aco.ElitistAntSystem._postprocess_iteration
.. automethod:: culebra.trainer.aco.ElitistAntSystem._finish_iteration
.. automethod:: culebra.trainer.aco.ElitistAntSystem._do_iteration_stats
.. automethod:: culebra.trainer.aco.ElitistAntSystem._default_termination_func
.. automethod:: culebra.trainer.aco.ElitistAntSystem._termination_criterion
.. automethod:: culebra.trainer.aco.ElitistAntSystem._init_representatives
.. automethod:: culebra.trainer.aco.ElitistAntSystem._set_cooperative_fitness
.. automethod:: culebra.trainer.aco.ElitistAntSystem._calculate_choice_info
.. automethod:: culebra.trainer.aco.ElitistAntSystem._initial_choice
.. automethod:: culebra.trainer.aco.ElitistAntSystem._next_choice
.. automethod:: culebra.trainer.aco.ElitistAntSystem._generate_ant
.. automethod:: culebra.trainer.aco.ElitistAntSystem._generate_col
.. automethod:: culebra.trainer.aco.ElitistAntSystem._init_pheromone
.. automethod:: culebra.trainer.aco.ElitistAntSystem._deposit_pheromone
.. automethod:: culebra.trainer.aco.ElitistAntSystem._should_reset_pheromone
.. automethod:: culebra.trainer.aco.ElitistAntSystem._increase_pheromone
.. automethod:: culebra.trainer.aco.ElitistAntSystem._decrease_pheromone
.. automethod:: culebra.trainer.aco.ElitistAntSystem._update_pheromone
.. automethod:: culebra.trainer.aco.ElitistAntSystem._update_elite
.. automethod:: culebra.trainer.aco.ElitistAntSystem._has_converged
