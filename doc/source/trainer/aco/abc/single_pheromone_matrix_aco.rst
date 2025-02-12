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

:py:class:`culebra.trainer.aco.abc.SinglePheromoneMatrixACO` class
==================================================================

.. autoclass:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO

Class attributes
----------------
.. autoattribute:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO.stats_names
.. autoattribute:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO.objective_stats

Class methods
-------------
.. automethod:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO.load_pickle

Properties
----------
.. autoproperty:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO.num_pheromone_matrices
.. autoproperty:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO.num_heuristic_matrices
.. autoproperty:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO.solution_cls
.. autoproperty:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO.species
.. autoproperty:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO.fitness_function
.. autoproperty:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO.initial_pheromone
.. autoproperty:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO.heuristic
.. autoproperty:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO.pheromone_influence
.. autoproperty:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO.heuristic_influence
.. autoproperty:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO.pheromone
.. autoproperty:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO.choice_info
.. autoproperty:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO.max_num_iters
.. autoproperty:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO.current_iter
.. autoproperty:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO.custom_termination_func
.. autoproperty:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO.col_size
.. autoproperty:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO.checkpoint_enable
.. autoproperty:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO.checkpoint_freq
.. autoproperty:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO.checkpoint_filename
.. autoproperty:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO.verbose
.. autoproperty:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO.random_seed
.. autoproperty:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO.logbook
.. autoproperty:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO.num_evals
.. autoproperty:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO.runtime
.. autoproperty:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO.index
.. autoproperty:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO.container
.. autoproperty:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO.representatives
.. autoproperty:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO.col

Methods
-------
.. automethod:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO.save_pickle
.. automethod:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO.reset
.. automethod:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO.evaluate
.. automethod:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO.best_solutions
.. automethod:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO.best_representatives
.. automethod:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO.train
.. automethod:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO.test

Private methods
---------------
.. automethod:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO._get_state
.. automethod:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO._set_state
.. automethod:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO._save_state
.. automethod:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO._load_state
.. automethod:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO._new_state
.. automethod:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO._init_state
.. automethod:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO._reset_state
.. automethod:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO._init_internals
.. automethod:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO._reset_internals
.. automethod:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO._init_search
.. automethod:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO._search
.. automethod:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO._finish_search
.. automethod:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO._start_iteration
.. automethod:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO._preprocess_iteration
.. automethod:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO._do_iteration
.. automethod:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO._postprocess_iteration
.. automethod:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO._finish_iteration
.. automethod:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO._do_iteration_stats
.. automethod:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO._default_termination_func
.. automethod:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO._termination_criterion
.. automethod:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO._init_representatives
.. automethod:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO._calculate_choice_info
.. automethod:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO._initial_choice
.. automethod:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO._next_choice
.. automethod:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO._feasible_neighborhood_probs
.. automethod:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO._generate_ant
.. automethod:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO._generate_col
.. automethod:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO._init_pheromone
.. automethod:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO._deposit_pheromone
.. automethod:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO._increase_pheromone
.. automethod:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO._decrease_pheromone
.. automethod:: culebra.trainer.aco.abc.SinglePheromoneMatrixACO._update_pheromone

