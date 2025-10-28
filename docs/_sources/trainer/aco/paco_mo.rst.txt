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

:py:class:`culebra.trainer.aco.PACO_MO` class
=============================================

.. autoclass:: culebra.trainer.aco.PACO_MO

Class attributes
----------------
.. autoattribute:: culebra.trainer.aco.PACO_MO.stats_names
.. autoattribute:: culebra.trainer.aco.PACO_MO.objective_stats

Class methods
-------------
.. automethod:: culebra.trainer.aco.PACO_MO.load

Properties
----------
.. autoproperty:: culebra.trainer.aco.PACO_MO.num_pheromone_matrices
.. autoproperty:: culebra.trainer.aco.PACO_MO.num_heuristic_matrices
.. autoproperty:: culebra.trainer.aco.PACO_MO.solution_cls
.. autoproperty:: culebra.trainer.aco.PACO_MO.species
.. autoproperty:: culebra.trainer.aco.PACO_MO.fitness_function
.. autoproperty:: culebra.trainer.aco.PACO_MO.initial_pheromone
.. autoproperty:: culebra.trainer.aco.PACO_MO.max_pheromone
.. autoproperty:: culebra.trainer.aco.PACO_MO.heuristic
.. autoproperty:: culebra.trainer.aco.PACO_MO.pheromone_influence
.. autoproperty:: culebra.trainer.aco.PACO_MO.heuristic_influence
.. autoproperty:: culebra.trainer.aco.PACO_MO.exploitation_prob
.. autoproperty:: culebra.trainer.aco.PACO_MO.pheromone
.. autoproperty:: culebra.trainer.aco.PACO_MO.choice_info
.. autoproperty:: culebra.trainer.aco.PACO_MO.max_num_iters
.. autoproperty:: culebra.trainer.aco.PACO_MO.current_iter
.. autoproperty:: culebra.trainer.aco.PACO_MO.custom_termination_func
.. autoproperty:: culebra.trainer.aco.PACO_MO.col_size
.. autoproperty:: culebra.trainer.aco.PACO_MO.pop_size
.. autoproperty:: culebra.trainer.aco.PACO_MO.checkpoint_enable
.. autoproperty:: culebra.trainer.aco.PACO_MO.checkpoint_freq
.. autoproperty:: culebra.trainer.aco.PACO_MO.checkpoint_filename
.. autoproperty:: culebra.trainer.aco.PACO_MO.verbose
.. autoproperty:: culebra.trainer.aco.PACO_MO.random_seed
.. autoproperty:: culebra.trainer.aco.PACO_MO.logbook
.. autoproperty:: culebra.trainer.aco.PACO_MO.num_evals
.. autoproperty:: culebra.trainer.aco.PACO_MO.runtime
.. autoproperty:: culebra.trainer.aco.PACO_MO.index
.. autoproperty:: culebra.trainer.aco.PACO_MO.container
.. autoproperty:: culebra.trainer.aco.PACO_MO.representatives
.. autoproperty:: culebra.trainer.aco.PACO_MO.col
.. autoproperty:: culebra.trainer.aco.PACO_MO.pop

Methods
-------
.. automethod:: culebra.trainer.aco.PACO_MO.dump
.. automethod:: culebra.trainer.aco.PACO_MO.reset
.. automethod:: culebra.trainer.aco.PACO_MO.evaluate
.. automethod:: culebra.trainer.aco.PACO_MO.best_solutions
.. automethod:: culebra.trainer.aco.PACO_MO.best_representatives
.. automethod:: culebra.trainer.aco.PACO_MO.train
.. automethod:: culebra.trainer.aco.PACO_MO.test

Private methods
---------------
.. automethod:: culebra.trainer.aco.PACO_MO._get_state
.. automethod:: culebra.trainer.aco.PACO_MO._set_state
.. automethod:: culebra.trainer.aco.PACO_MO._save_state
.. automethod:: culebra.trainer.aco.PACO_MO._load_state
.. automethod:: culebra.trainer.aco.PACO_MO._new_state
.. automethod:: culebra.trainer.aco.PACO_MO._init_state
.. automethod:: culebra.trainer.aco.PACO_MO._reset_state
.. automethod:: culebra.trainer.aco.PACO_MO._init_internals
.. automethod:: culebra.trainer.aco.PACO_MO._reset_internals
.. automethod:: culebra.trainer.aco.PACO_MO._init_search
.. automethod:: culebra.trainer.aco.PACO_MO._search
.. automethod:: culebra.trainer.aco.PACO_MO._finish_search
.. automethod:: culebra.trainer.aco.PACO_MO._start_iteration
.. automethod:: culebra.trainer.aco.PACO_MO._preprocess_iteration
.. automethod:: culebra.trainer.aco.PACO_MO._do_iteration
.. automethod:: culebra.trainer.aco.PACO_MO._postprocess_iteration
.. automethod:: culebra.trainer.aco.PACO_MO._finish_iteration
.. automethod:: culebra.trainer.aco.PACO_MO._do_iteration_stats
.. automethod:: culebra.trainer.aco.PACO_MO._default_termination_func
.. automethod:: culebra.trainer.aco.PACO_MO._termination_criterion
.. automethod:: culebra.trainer.aco.PACO_MO._init_representatives
.. automethod:: culebra.trainer.aco.PACO_MO._set_cooperative_fitness
.. automethod:: culebra.trainer.aco.PACO_MO._calculate_choice_info
.. automethod:: culebra.trainer.aco.PACO_MO._initial_choice
.. automethod:: culebra.trainer.aco.PACO_MO._next_choice
.. automethod:: culebra.trainer.aco.PACO_MO._generate_ant
.. automethod:: culebra.trainer.aco.PACO_MO._generate_col
.. automethod:: culebra.trainer.aco.PACO_MO._init_pheromone
.. automethod:: culebra.trainer.aco.PACO_MO._deposit_pheromone
.. automethod:: culebra.trainer.aco.PACO_MO._increase_pheromone
.. automethod:: culebra.trainer.aco.PACO_MO._decrease_pheromone
.. automethod:: culebra.trainer.aco.PACO_MO._update_pheromone
.. automethod:: culebra.trainer.aco.PACO_MO._update_pop
.. automethod:: culebra.trainer.aco.PACO_MO._update_elite
