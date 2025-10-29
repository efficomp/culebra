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

:py:class:`culebra.trainer.aco.abc.PheromoneBasedACO` class
===========================================================

.. autoclass:: culebra.trainer.aco.abc.PheromoneBasedACO

Class attributes
----------------
.. autoattribute:: culebra.trainer.aco.abc.PheromoneBasedACO.stats_names
.. autoattribute:: culebra.trainer.aco.abc.PheromoneBasedACO.objective_stats

Class methods
-------------
.. automethod:: culebra.trainer.aco.abc.PheromoneBasedACO.load

Properties
----------
.. autoproperty:: culebra.trainer.aco.abc.PheromoneBasedACO.num_pheromone_matrices
.. autoproperty:: culebra.trainer.aco.abc.PheromoneBasedACO.num_heuristic_matrices
.. autoproperty:: culebra.trainer.aco.abc.PheromoneBasedACO.solution_cls
.. autoproperty:: culebra.trainer.aco.abc.PheromoneBasedACO.species
.. autoproperty:: culebra.trainer.aco.abc.PheromoneBasedACO.fitness_function
.. autoproperty:: culebra.trainer.aco.abc.PheromoneBasedACO.initial_pheromone
.. autoproperty:: culebra.trainer.aco.abc.PheromoneBasedACO.heuristic
.. autoproperty:: culebra.trainer.aco.abc.PheromoneBasedACO.pheromone_influence
.. autoproperty:: culebra.trainer.aco.abc.PheromoneBasedACO.heuristic_influence
.. autoproperty:: culebra.trainer.aco.abc.PheromoneBasedACO.exploitation_prob
.. autoproperty:: culebra.trainer.aco.abc.PheromoneBasedACO.pheromone_evaporation_rate
.. autoproperty:: culebra.trainer.aco.abc.PheromoneBasedACO.pheromone
.. autoproperty:: culebra.trainer.aco.abc.PheromoneBasedACO.choice_info
.. autoproperty:: culebra.trainer.aco.abc.PheromoneBasedACO.max_num_iters
.. autoproperty:: culebra.trainer.aco.abc.PheromoneBasedACO.current_iter
.. autoproperty:: culebra.trainer.aco.abc.PheromoneBasedACO.custom_termination_func
.. autoproperty:: culebra.trainer.aco.abc.PheromoneBasedACO.col_size
.. autoproperty:: culebra.trainer.aco.abc.PheromoneBasedACO.checkpoint_enable
.. autoproperty:: culebra.trainer.aco.abc.PheromoneBasedACO.checkpoint_freq
.. autoproperty:: culebra.trainer.aco.abc.PheromoneBasedACO.checkpoint_filename
.. autoproperty:: culebra.trainer.aco.abc.PheromoneBasedACO.verbose
.. autoproperty:: culebra.trainer.aco.abc.PheromoneBasedACO.random_seed
.. autoproperty:: culebra.trainer.aco.abc.PheromoneBasedACO.logbook
.. autoproperty:: culebra.trainer.aco.abc.PheromoneBasedACO.num_evals
.. autoproperty:: culebra.trainer.aco.abc.PheromoneBasedACO.runtime
.. autoproperty:: culebra.trainer.aco.abc.PheromoneBasedACO.index
.. autoproperty:: culebra.trainer.aco.abc.PheromoneBasedACO.container
.. autoproperty:: culebra.trainer.aco.abc.PheromoneBasedACO.representatives
.. autoproperty:: culebra.trainer.aco.abc.PheromoneBasedACO.col

Methods
-------
.. automethod:: culebra.trainer.aco.abc.PheromoneBasedACO.dump
.. automethod:: culebra.trainer.aco.abc.PheromoneBasedACO.reset
.. automethod:: culebra.trainer.aco.abc.PheromoneBasedACO.evaluate
.. automethod:: culebra.trainer.aco.abc.PheromoneBasedACO.best_solutions
.. automethod:: culebra.trainer.aco.abc.PheromoneBasedACO.best_representatives
.. automethod:: culebra.trainer.aco.abc.PheromoneBasedACO.train
.. automethod:: culebra.trainer.aco.abc.PheromoneBasedACO.test

Private methods
---------------
.. automethod:: culebra.trainer.aco.abc.PheromoneBasedACO._get_state
.. automethod:: culebra.trainer.aco.abc.PheromoneBasedACO._set_state
.. automethod:: culebra.trainer.aco.abc.PheromoneBasedACO._save_state
.. automethod:: culebra.trainer.aco.abc.PheromoneBasedACO._load_state
.. automethod:: culebra.trainer.aco.abc.PheromoneBasedACO._new_state
.. automethod:: culebra.trainer.aco.abc.PheromoneBasedACO._init_state
.. automethod:: culebra.trainer.aco.abc.PheromoneBasedACO._reset_state
.. automethod:: culebra.trainer.aco.abc.PheromoneBasedACO._init_internals
.. automethod:: culebra.trainer.aco.abc.PheromoneBasedACO._reset_internals
.. automethod:: culebra.trainer.aco.abc.PheromoneBasedACO._init_search
.. automethod:: culebra.trainer.aco.abc.PheromoneBasedACO._search
.. automethod:: culebra.trainer.aco.abc.PheromoneBasedACO._finish_search
.. automethod:: culebra.trainer.aco.abc.PheromoneBasedACO._start_iteration
.. automethod:: culebra.trainer.aco.abc.PheromoneBasedACO._preprocess_iteration
.. automethod:: culebra.trainer.aco.abc.PheromoneBasedACO._do_iteration
.. automethod:: culebra.trainer.aco.abc.PheromoneBasedACO._postprocess_iteration
.. automethod:: culebra.trainer.aco.abc.PheromoneBasedACO._finish_iteration
.. automethod:: culebra.trainer.aco.abc.PheromoneBasedACO._do_iteration_stats
.. automethod:: culebra.trainer.aco.abc.PheromoneBasedACO._default_termination_func
.. automethod:: culebra.trainer.aco.abc.PheromoneBasedACO._termination_criterion
.. automethod:: culebra.trainer.aco.abc.PheromoneBasedACO._init_representatives
.. automethod:: culebra.trainer.aco.abc.PheromoneBasedACO._set_cooperative_fitness
.. automethod:: culebra.trainer.aco.abc.PheromoneBasedACO._calculate_choice_info
.. automethod:: culebra.trainer.aco.abc.PheromoneBasedACO._initial_choice
.. automethod:: culebra.trainer.aco.abc.PheromoneBasedACO._next_choice
.. automethod:: culebra.trainer.aco.abc.PheromoneBasedACO._generate_ant
.. automethod:: culebra.trainer.aco.abc.PheromoneBasedACO._generate_col
.. automethod:: culebra.trainer.aco.abc.PheromoneBasedACO._init_pheromone
.. automethod:: culebra.trainer.aco.abc.PheromoneBasedACO._pheromone_amount
.. automethod:: culebra.trainer.aco.abc.PheromoneBasedACO._deposit_pheromone
.. automethod:: culebra.trainer.aco.abc.PheromoneBasedACO._increase_pheromone
.. automethod:: culebra.trainer.aco.abc.PheromoneBasedACO._decrease_pheromone
.. automethod:: culebra.trainer.aco.abc.PheromoneBasedACO._update_pheromone

