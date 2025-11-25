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

:class:`culebra.trainer.aco.PACOFS` class
=========================================

.. autoclass:: culebra.trainer.aco.PACOFS

Class attributes
----------------
.. autoattribute:: culebra.trainer.aco.PACOFS.objective_stats
.. autoattribute:: culebra.trainer.aco.PACOFS.stats_names

Class methods
-------------
.. automethod:: culebra.trainer.aco.PACOFS.load

Properties
----------
.. autoproperty:: culebra.trainer.aco.PACOFS.checkpoint_enable
.. autoproperty:: culebra.trainer.aco.PACOFS.checkpoint_filename
.. autoproperty:: culebra.trainer.aco.PACOFS.checkpoint_freq
.. autoproperty:: culebra.trainer.aco.PACOFS.choice_info
.. autoproperty:: culebra.trainer.aco.PACOFS.col
.. autoproperty:: culebra.trainer.aco.PACOFS.col_size
.. autoproperty:: culebra.trainer.aco.PACOFS.container
.. autoproperty:: culebra.trainer.aco.PACOFS.current_iter
.. autoproperty:: culebra.trainer.aco.PACOFS.custom_termination_func
.. autoproperty:: culebra.trainer.aco.PACOFS.discard_prob
.. autoproperty:: culebra.trainer.aco.PACOFS.exploitation_prob
.. autoproperty:: culebra.trainer.aco.PACOFS.fitness_function
.. autoproperty:: culebra.trainer.aco.PACOFS.heuristic
.. autoproperty:: culebra.trainer.aco.PACOFS.heuristic_influence
.. autoproperty:: culebra.trainer.aco.PACOFS.heuristic_shapes
.. autoproperty:: culebra.trainer.aco.PACOFS.index
.. autoproperty:: culebra.trainer.aco.PACOFS.initial_pheromone
.. autoproperty:: culebra.trainer.aco.PACOFS.logbook
.. autoproperty:: culebra.trainer.aco.PACOFS.max_num_iters
.. autoproperty:: culebra.trainer.aco.PACOFS.num_evals
.. autoproperty:: culebra.trainer.aco.PACOFS.num_heuristic_matrices
.. autoproperty:: culebra.trainer.aco.PACOFS.num_pheromone_matrices
.. autoproperty:: culebra.trainer.aco.PACOFS.pheromone
.. autoproperty:: culebra.trainer.aco.PACOFS.pheromone_influence
.. autoproperty:: culebra.trainer.aco.PACOFS.pheromone_shapes
.. autoproperty:: culebra.trainer.aco.PACOFS.pop
.. autoproperty:: culebra.trainer.aco.PACOFS.pop_size
.. autoproperty:: culebra.trainer.aco.PACOFS.random_seed
.. autoproperty:: culebra.trainer.aco.PACOFS.representatives
.. autoproperty:: culebra.trainer.aco.PACOFS.runtime
.. autoproperty:: culebra.trainer.aco.PACOFS.solution_cls
.. autoproperty:: culebra.trainer.aco.PACOFS.species
.. autoproperty:: culebra.trainer.aco.PACOFS.verbose

Methods
-------
.. automethod:: culebra.trainer.aco.PACOFS.best_representatives
.. automethod:: culebra.trainer.aco.PACOFS.best_solutions
.. automethod:: culebra.trainer.aco.PACOFS.dump
.. automethod:: culebra.trainer.aco.PACOFS.evaluate
.. automethod:: culebra.trainer.aco.PACOFS.reset
.. automethod:: culebra.trainer.aco.PACOFS.test
.. automethod:: culebra.trainer.aco.PACOFS.train

Private methods
---------------
.. automethod:: culebra.trainer.aco.PACOFS._ant_choice_info
.. automethod:: culebra.trainer.aco.PACOFS._calculate_choice_info
.. automethod:: culebra.trainer.aco.PACOFS._decrease_pheromone
.. automethod:: culebra.trainer.aco.PACOFS._default_termination_func
.. automethod:: culebra.trainer.aco.PACOFS._deposit_pheromone
.. automethod:: culebra.trainer.aco.PACOFS._do_iteration
.. automethod:: culebra.trainer.aco.PACOFS._do_iteration_stats
.. automethod:: culebra.trainer.aco.PACOFS._finish_iteration
.. automethod:: culebra.trainer.aco.PACOFS._finish_search
.. automethod:: culebra.trainer.aco.PACOFS._generate_ant
.. automethod:: culebra.trainer.aco.PACOFS._generate_col
.. automethod:: culebra.trainer.aco.PACOFS._get_state
.. automethod:: culebra.trainer.aco.PACOFS._increase_pheromone
.. automethod:: culebra.trainer.aco.PACOFS._init_internals
.. automethod:: culebra.trainer.aco.PACOFS._init_pheromone
.. automethod:: culebra.trainer.aco.PACOFS._init_representatives
.. automethod:: culebra.trainer.aco.PACOFS._init_search
.. automethod:: culebra.trainer.aco.PACOFS._init_state
.. automethod:: culebra.trainer.aco.PACOFS._load_state
.. automethod:: culebra.trainer.aco.PACOFS._new_state
.. automethod:: culebra.trainer.aco.PACOFS._next_choice
.. automethod:: culebra.trainer.aco.PACOFS._pheromone_amount
.. automethod:: culebra.trainer.aco.PACOFS._postprocess_iteration
.. automethod:: culebra.trainer.aco.PACOFS._preprocess_iteration
.. automethod:: culebra.trainer.aco.PACOFS._reset_internals
.. automethod:: culebra.trainer.aco.PACOFS._reset_state
.. automethod:: culebra.trainer.aco.PACOFS._save_state
.. automethod:: culebra.trainer.aco.PACOFS._search
.. automethod:: culebra.trainer.aco.PACOFS._set_cooperative_fitness
.. automethod:: culebra.trainer.aco.PACOFS._set_state
.. automethod:: culebra.trainer.aco.PACOFS._start_iteration
.. automethod:: culebra.trainer.aco.PACOFS._termination_criterion
.. automethod:: culebra.trainer.aco.PACOFS._update_pheromone
.. automethod:: culebra.trainer.aco.PACOFS._update_pop
