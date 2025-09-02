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

:py:class:`culebra.trainer.aco.abc.MultipleHeuristicMatricesACO` class
======================================================================

.. autoclass:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO

Class attributes
----------------
.. autoattribute:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO.stats_names
.. autoattribute:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO.objective_stats

Class methods
-------------
.. automethod:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO.load

Properties
----------
.. autoproperty:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO.num_pheromone_matrices
.. autoproperty:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO.num_heuristic_matrices
.. autoproperty:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO.solution_cls
.. autoproperty:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO.species
.. autoproperty:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO.fitness_function
.. autoproperty:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO.initial_pheromone
.. autoproperty:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO.heuristic
.. autoproperty:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO.pheromone_influence
.. autoproperty:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO.heuristic_influence
.. autoproperty:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO.pheromone
.. autoproperty:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO.choice_info
.. autoproperty:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO.max_num_iters
.. autoproperty:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO.current_iter
.. autoproperty:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO.custom_termination_func
.. autoproperty:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO.col_size
.. autoproperty:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO.checkpoint_enable
.. autoproperty:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO.checkpoint_freq
.. autoproperty:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO.checkpoint_filename
.. autoproperty:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO.verbose
.. autoproperty:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO.random_seed
.. autoproperty:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO.logbook
.. autoproperty:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO.num_evals
.. autoproperty:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO.runtime
.. autoproperty:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO.index
.. autoproperty:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO.container
.. autoproperty:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO.representatives
.. autoproperty:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO.col

Methods
-------
.. automethod:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO.dump
.. automethod:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO.reset
.. automethod:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO.evaluate
.. automethod:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO.best_solutions
.. automethod:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO.best_representatives
.. automethod:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO.train
.. automethod:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO.test

Private methods
---------------
.. automethod:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO._get_state
.. automethod:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO._set_state
.. automethod:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO._save_state
.. automethod:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO._load_state
.. automethod:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO._new_state
.. automethod:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO._init_state
.. automethod:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO._reset_state
.. automethod:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO._init_internals
.. automethod:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO._reset_internals
.. automethod:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO._init_search
.. automethod:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO._search
.. automethod:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO._finish_search
.. automethod:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO._start_iteration
.. automethod:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO._preprocess_iteration
.. automethod:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO._do_iteration
.. automethod:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO._postprocess_iteration
.. automethod:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO._finish_iteration
.. automethod:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO._do_iteration_stats
.. automethod:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO._default_termination_func
.. automethod:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO._termination_criterion
.. automethod:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO._init_representatives
.. automethod:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO._calculate_choice_info
.. automethod:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO._initial_choice
.. automethod:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO._next_choice
.. automethod:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO._feasible_neighborhood_probs
.. automethod:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO._generate_ant
.. automethod:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO._generate_col
.. automethod:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO._init_pheromone
.. automethod:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO._deposit_pheromone
.. automethod:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO._increase_pheromone
.. automethod:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO._decrease_pheromone
.. automethod:: culebra.trainer.aco.abc.MultipleHeuristicMatricesACO._update_pheromone

