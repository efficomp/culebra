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

:py:class:`culebra.trainer.aco.abc.MultiplePheromoneMatricesACO` class
======================================================================

.. autoclass:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO

Class attributes
----------------
.. autoattribute:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO.stats_names
.. autoattribute:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO.objective_stats

Class methods
-------------
.. automethod:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO.load

Properties
----------
.. autoproperty:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO.num_pheromone_matrices
.. autoproperty:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO.num_heuristic_matrices
.. autoproperty:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO.solution_cls
.. autoproperty:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO.species
.. autoproperty:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO.fitness_function
.. autoproperty:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO.initial_pheromone
.. autoproperty:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO.heuristic
.. autoproperty:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO.pheromone_influence
.. autoproperty:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO.heuristic_influence
.. autoproperty:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO.exploitation_prob
.. autoproperty:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO.pheromone
.. autoproperty:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO.choice_info
.. autoproperty:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO.max_num_iters
.. autoproperty:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO.current_iter
.. autoproperty:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO.custom_termination_func
.. autoproperty:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO.col_size
.. autoproperty:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO.checkpoint_enable
.. autoproperty:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO.checkpoint_freq
.. autoproperty:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO.checkpoint_filename
.. autoproperty:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO.verbose
.. autoproperty:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO.random_seed
.. autoproperty:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO.logbook
.. autoproperty:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO.num_evals
.. autoproperty:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO.runtime
.. autoproperty:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO.index
.. autoproperty:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO.container
.. autoproperty:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO.representatives
.. autoproperty:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO.col

Methods
-------
.. automethod:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO.dump
.. automethod:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO.reset
.. automethod:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO.evaluate
.. automethod:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO.best_solutions
.. automethod:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO.best_representatives
.. automethod:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO.train
.. automethod:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO.test

Private methods
---------------
.. automethod:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO._get_state
.. automethod:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO._set_state
.. automethod:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO._save_state
.. automethod:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO._load_state
.. automethod:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO._new_state
.. automethod:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO._init_state
.. automethod:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO._reset_state
.. automethod:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO._init_internals
.. automethod:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO._reset_internals
.. automethod:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO._init_search
.. automethod:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO._search
.. automethod:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO._finish_search
.. automethod:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO._start_iteration
.. automethod:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO._preprocess_iteration
.. automethod:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO._do_iteration
.. automethod:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO._postprocess_iteration
.. automethod:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO._finish_iteration
.. automethod:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO._do_iteration_stats
.. automethod:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO._default_termination_func
.. automethod:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO._termination_criterion
.. automethod:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO._init_representatives
.. automethod:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO._set_cooperative_fitness
.. automethod:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO._calculate_choice_info
.. automethod:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO._initial_choice
.. automethod:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO._next_choice
.. automethod:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO._generate_ant
.. automethod:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO._generate_col
.. automethod:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO._init_pheromone
.. automethod:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO._pheromone_amount
.. automethod:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO._deposit_pheromone
.. automethod:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO._increase_pheromone
.. automethod:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO._decrease_pheromone
.. automethod:: culebra.trainer.aco.abc.MultiplePheromoneMatricesACO._update_pheromone

