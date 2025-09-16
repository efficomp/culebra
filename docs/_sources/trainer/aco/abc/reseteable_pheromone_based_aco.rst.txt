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

:py:class:`culebra.trainer.aco.abc.ReseteablePheromoneBasedACO` class
=====================================================================

.. autoclass:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO

Class attributes
----------------
.. autoattribute:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO.stats_names
.. autoattribute:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO.objective_stats

Class methods
-------------
.. automethod:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO.load

Properties
----------
.. autoproperty:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO.num_pheromone_matrices
.. autoproperty:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO.num_heuristic_matrices
.. autoproperty:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO.solution_cls
.. autoproperty:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO.species
.. autoproperty:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO.fitness_function
.. autoproperty:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO.initial_pheromone
.. autoproperty:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO.heuristic
.. autoproperty:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO.pheromone_influence
.. autoproperty:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO.heuristic_influence
.. autoproperty:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO.pheromone
.. autoproperty:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO.choice_info
.. autoproperty:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO.convergence_check_freq
.. autoproperty:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO.max_num_iters
.. autoproperty:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO.current_iter
.. autoproperty:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO.custom_termination_func
.. autoproperty:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO.col_size
.. autoproperty:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO.checkpoint_enable
.. autoproperty:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO.checkpoint_freq
.. autoproperty:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO.checkpoint_filename
.. autoproperty:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO.verbose
.. autoproperty:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO.random_seed
.. autoproperty:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO.logbook
.. autoproperty:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO.num_evals
.. autoproperty:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO.runtime
.. autoproperty:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO.index
.. autoproperty:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO.container
.. autoproperty:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO.representatives
.. autoproperty:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO.col

Methods
-------
.. automethod:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO.dump
.. automethod:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO.reset
.. automethod:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO.evaluate
.. automethod:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO.best_solutions
.. automethod:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO.best_representatives
.. automethod:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO.train
.. automethod:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO.test

Private methods
---------------
.. automethod:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO._get_state
.. automethod:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO._set_state
.. automethod:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO._save_state
.. automethod:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO._load_state
.. automethod:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO._new_state
.. automethod:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO._init_state
.. automethod:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO._reset_state
.. automethod:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO._init_internals
.. automethod:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO._reset_internals
.. automethod:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO._init_search
.. automethod:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO._search
.. automethod:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO._finish_search
.. automethod:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO._start_iteration
.. automethod:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO._preprocess_iteration
.. automethod:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO._do_iteration
.. automethod:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO._postprocess_iteration
.. automethod:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO._finish_iteration
.. automethod:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO._do_iteration_stats
.. automethod:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO._default_termination_func
.. automethod:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO._termination_criterion
.. automethod:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO._init_representatives
.. automethod:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO._set_cooperative_fitness
.. automethod:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO._calculate_choice_info
.. automethod:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO._initial_choice
.. automethod:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO._next_choice
.. automethod:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO._feasible_neighborhood_probs
.. automethod:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO._generate_ant
.. automethod:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO._generate_col
.. automethod:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO._init_pheromone
.. automethod:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO._deposit_pheromone
.. automethod:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO._should_reset_pheromone
.. automethod:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO._increase_pheromone
.. automethod:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO._decrease_pheromone
.. automethod:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO._update_pheromone
.. automethod:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO._update_elite
.. automethod:: culebra.trainer.aco.abc.ReseteablePheromoneBasedACO._has_converged
