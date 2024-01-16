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

:py:class:`culebra.trainer.aco.abc.SingleObjPACO` class
=======================================================

.. autoclass:: culebra.trainer.aco.abc.SingleObjPACO

Class attributes
----------------
.. autoattribute:: culebra.trainer.aco.abc.SingleObjPACO.stats_names
.. autoattribute:: culebra.trainer.aco.abc.SingleObjPACO.objective_stats

Properties
----------
.. autoproperty:: culebra.trainer.aco.abc.SingleObjPACO.num_pheromone_matrices
.. autoproperty:: culebra.trainer.aco.abc.SingleObjPACO.num_heuristic_matrices
.. autoproperty:: culebra.trainer.aco.abc.SingleObjPACO.solution_cls
.. autoproperty:: culebra.trainer.aco.abc.SingleObjPACO.species
.. autoproperty:: culebra.trainer.aco.abc.SingleObjPACO.fitness_function
.. autoproperty:: culebra.trainer.aco.abc.SingleObjPACO.initial_pheromone
.. autoproperty:: culebra.trainer.aco.abc.SingleObjPACO.max_pheromone
.. autoproperty:: culebra.trainer.aco.abc.SingleObjPACO.heuristic
.. autoproperty:: culebra.trainer.aco.abc.SingleObjPACO.pheromone_influence
.. autoproperty:: culebra.trainer.aco.abc.SingleObjPACO.heuristic_influence
.. autoproperty:: culebra.trainer.aco.abc.SingleObjPACO.pheromone
.. autoproperty:: culebra.trainer.aco.abc.SingleObjPACO.choice_info
.. autoproperty:: culebra.trainer.aco.abc.SingleObjPACO.max_num_iters
.. autoproperty:: culebra.trainer.aco.abc.SingleObjPACO.current_iter
.. autoproperty:: culebra.trainer.aco.abc.SingleObjPACO.custom_termination_func
.. autoproperty:: culebra.trainer.aco.abc.SingleObjPACO.col_size
.. autoproperty:: culebra.trainer.aco.abc.SingleObjPACO.pop_size
.. autoproperty:: culebra.trainer.aco.abc.SingleObjPACO.checkpoint_enable
.. autoproperty:: culebra.trainer.aco.abc.SingleObjPACO.checkpoint_freq
.. autoproperty:: culebra.trainer.aco.abc.SingleObjPACO.checkpoint_filename
.. autoproperty:: culebra.trainer.aco.abc.SingleObjPACO.verbose
.. autoproperty:: culebra.trainer.aco.abc.SingleObjPACO.random_seed
.. autoproperty:: culebra.trainer.aco.abc.SingleObjPACO.logbook
.. autoproperty:: culebra.trainer.aco.abc.SingleObjPACO.num_evals
.. autoproperty:: culebra.trainer.aco.abc.SingleObjPACO.runtime
.. autoproperty:: culebra.trainer.aco.abc.SingleObjPACO.index
.. autoproperty:: culebra.trainer.aco.abc.SingleObjPACO.container
.. autoproperty:: culebra.trainer.aco.abc.SingleObjPACO.representatives
.. autoproperty:: culebra.trainer.aco.abc.SingleObjPACO.col
.. autoproperty:: culebra.trainer.aco.abc.SingleObjPACO.pop

Methods
-------
.. automethod:: culebra.trainer.aco.abc.SingleObjPACO.reset
.. automethod:: culebra.trainer.aco.abc.SingleObjPACO.evaluate
.. automethod:: culebra.trainer.aco.abc.SingleObjPACO.best_solutions
.. automethod:: culebra.trainer.aco.abc.SingleObjPACO.best_representatives
.. automethod:: culebra.trainer.aco.abc.SingleObjPACO.train
.. automethod:: culebra.trainer.aco.abc.SingleObjPACO.test

Private methods
---------------
.. automethod:: culebra.trainer.aco.abc.SingleObjPACO._get_state
.. automethod:: culebra.trainer.aco.abc.SingleObjPACO._set_state
.. automethod:: culebra.trainer.aco.abc.SingleObjPACO._save_state
.. automethod:: culebra.trainer.aco.abc.SingleObjPACO._load_state
.. automethod:: culebra.trainer.aco.abc.SingleObjPACO._new_state
.. automethod:: culebra.trainer.aco.abc.SingleObjPACO._init_state
.. automethod:: culebra.trainer.aco.abc.SingleObjPACO._reset_state
.. automethod:: culebra.trainer.aco.abc.SingleObjPACO._init_internals
.. automethod:: culebra.trainer.aco.abc.SingleObjPACO._reset_internals
.. automethod:: culebra.trainer.aco.abc.SingleObjPACO._init_search
.. automethod:: culebra.trainer.aco.abc.SingleObjPACO._search
.. automethod:: culebra.trainer.aco.abc.SingleObjPACO._finish_search
.. automethod:: culebra.trainer.aco.abc.SingleObjPACO._start_iteration
.. automethod:: culebra.trainer.aco.abc.SingleObjPACO._preprocess_iteration
.. automethod:: culebra.trainer.aco.abc.SingleObjPACO._do_iteration
.. automethod:: culebra.trainer.aco.abc.SingleObjPACO._postprocess_iteration
.. automethod:: culebra.trainer.aco.abc.SingleObjPACO._finish_iteration
.. automethod:: culebra.trainer.aco.abc.SingleObjPACO._do_iteration_stats
.. automethod:: culebra.trainer.aco.abc.SingleObjPACO._default_termination_func
.. automethod:: culebra.trainer.aco.abc.SingleObjPACO._termination_criterion
.. automethod:: culebra.trainer.aco.abc.SingleObjPACO._init_representatives
.. automethod:: culebra.trainer.aco.abc.SingleObjPACO._calculate_choice_info
.. automethod:: culebra.trainer.aco.abc.SingleObjPACO._initial_choice
.. automethod:: culebra.trainer.aco.abc.SingleObjPACO._next_choice
.. automethod:: culebra.trainer.aco.abc.SingleObjPACO._feasible_neighborhood_probs
.. automethod:: culebra.trainer.aco.abc.SingleObjPACO._generate_ant
.. automethod:: culebra.trainer.aco.abc.SingleObjPACO._generate_col
.. automethod:: culebra.trainer.aco.abc.SingleObjPACO._init_pheromone
.. automethod:: culebra.trainer.aco.abc.SingleObjPACO._deposit_pheromone
.. automethod:: culebra.trainer.aco.abc.SingleObjPACO._increase_pheromone
.. automethod:: culebra.trainer.aco.abc.SingleObjPACO._decrease_pheromone
.. automethod:: culebra.trainer.aco.abc.SingleObjPACO._update_pheromone
.. automethod:: culebra.trainer.aco.abc.SingleObjPACO._update_pop

