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

:py:class:`culebra.trainer.aco.abc.SingleHeuristicMatrixACO` class
==================================================================

.. autoclass:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO

Class attributes
----------------
.. autoattribute:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO.stats_names
.. autoattribute:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO.objective_stats

Class methods
-------------
.. automethod:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO.load

Properties
----------
.. autoproperty:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO.num_pheromone_matrices
.. autoproperty:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO.num_heuristic_matrices
.. autoproperty:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO.solution_cls
.. autoproperty:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO.species
.. autoproperty:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO.fitness_function
.. autoproperty:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO.initial_pheromone
.. autoproperty:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO.heuristic
.. autoproperty:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO.pheromone_influence
.. autoproperty:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO.heuristic_influence
.. autoproperty:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO.pheromone
.. autoproperty:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO.choice_info
.. autoproperty:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO.max_num_iters
.. autoproperty:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO.current_iter
.. autoproperty:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO.custom_termination_func
.. autoproperty:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO.col_size
.. autoproperty:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO.checkpoint_enable
.. autoproperty:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO.checkpoint_freq
.. autoproperty:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO.checkpoint_filename
.. autoproperty:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO.verbose
.. autoproperty:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO.random_seed
.. autoproperty:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO.logbook
.. autoproperty:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO.num_evals
.. autoproperty:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO.runtime
.. autoproperty:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO.index
.. autoproperty:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO.container
.. autoproperty:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO.representatives
.. autoproperty:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO.col

Methods
-------
.. automethod:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO.dump
.. automethod:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO.reset
.. automethod:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO.evaluate
.. automethod:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO.best_solutions
.. automethod:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO.best_representatives
.. automethod:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO.train
.. automethod:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO.test

Private methods
---------------
.. automethod:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO._get_state
.. automethod:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO._set_state
.. automethod:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO._save_state
.. automethod:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO._load_state
.. automethod:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO._new_state
.. automethod:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO._init_state
.. automethod:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO._reset_state
.. automethod:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO._init_internals
.. automethod:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO._reset_internals
.. automethod:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO._init_search
.. automethod:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO._search
.. automethod:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO._finish_search
.. automethod:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO._start_iteration
.. automethod:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO._preprocess_iteration
.. automethod:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO._do_iteration
.. automethod:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO._postprocess_iteration
.. automethod:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO._finish_iteration
.. automethod:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO._do_iteration_stats
.. automethod:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO._default_termination_func
.. automethod:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO._termination_criterion
.. automethod:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO._init_representatives
.. automethod:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO._set_cooperative_fitness
.. automethod:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO._calculate_choice_info
.. automethod:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO._initial_choice
.. automethod:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO._next_choice
.. automethod:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO._feasible_neighborhood_probs
.. automethod:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO._generate_ant
.. automethod:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO._generate_col
.. automethod:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO._init_pheromone
.. automethod:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO._deposit_pheromone
.. automethod:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO._increase_pheromone
.. automethod:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO._decrease_pheromone
.. automethod:: culebra.trainer.aco.abc.SingleHeuristicMatrixACO._update_pheromone

