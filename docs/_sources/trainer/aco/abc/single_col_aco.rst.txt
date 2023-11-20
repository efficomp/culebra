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

:py:class:`culebra.trainer.aco.abc.SingleColACO` class
======================================================

.. autoclass:: culebra.trainer.aco.abc.SingleColACO

Class attributes
----------------
.. autoattribute:: culebra.trainer.aco.abc.SingleColACO.stats_names
.. autoattribute:: culebra.trainer.aco.abc.SingleColACO.objective_stats

Properties
----------
.. autoproperty:: culebra.trainer.aco.abc.SingleColACO.solution_cls
.. autoproperty:: culebra.trainer.aco.abc.SingleColACO.species
.. autoproperty:: culebra.trainer.aco.abc.SingleColACO.fitness_function
.. autoproperty:: culebra.trainer.aco.abc.SingleColACO.initial_pheromone
.. autoproperty:: culebra.trainer.aco.abc.SingleColACO.heuristic
.. autoproperty:: culebra.trainer.aco.abc.SingleColACO.pheromone_influence
.. autoproperty:: culebra.trainer.aco.abc.SingleColACO.heuristic_influence
.. autoproperty:: culebra.trainer.aco.abc.SingleColACO.pheromone
.. autoproperty:: culebra.trainer.aco.abc.SingleColACO.choice_info
.. autoproperty:: culebra.trainer.aco.abc.SingleColACO.max_num_iters
.. autoproperty:: culebra.trainer.aco.abc.SingleColACO.current_iter
.. autoproperty:: culebra.trainer.aco.abc.SingleColACO.custom_termination_func
.. autoproperty:: culebra.trainer.aco.abc.SingleColACO.col_size
.. autoproperty:: culebra.trainer.aco.abc.SingleColACO.checkpoint_enable
.. autoproperty:: culebra.trainer.aco.abc.SingleColACO.checkpoint_freq
.. autoproperty:: culebra.trainer.aco.abc.SingleColACO.checkpoint_filename
.. autoproperty:: culebra.trainer.aco.abc.SingleColACO.verbose
.. autoproperty:: culebra.trainer.aco.abc.SingleColACO.random_seed
.. autoproperty:: culebra.trainer.aco.abc.SingleColACO.logbook
.. autoproperty:: culebra.trainer.aco.abc.SingleColACO.num_evals
.. autoproperty:: culebra.trainer.aco.abc.SingleColACO.runtime
.. autoproperty:: culebra.trainer.aco.abc.SingleColACO.index
.. autoproperty:: culebra.trainer.aco.abc.SingleColACO.container
.. autoproperty:: culebra.trainer.aco.abc.SingleColACO.representatives
.. autoproperty:: culebra.trainer.aco.abc.SingleColACO.col

Methods
-------
.. automethod:: culebra.trainer.aco.abc.SingleColACO.reset
.. automethod:: culebra.trainer.aco.abc.SingleColACO.evaluate
.. automethod:: culebra.trainer.aco.abc.SingleColACO.best_solutions
.. automethod:: culebra.trainer.aco.abc.SingleColACO.best_representatives
.. automethod:: culebra.trainer.aco.abc.SingleColACO.train
.. automethod:: culebra.trainer.aco.abc.SingleColACO.test

Private methods
---------------
.. automethod:: culebra.trainer.aco.abc.SingleColACO._get_state
.. automethod:: culebra.trainer.aco.abc.SingleColACO._set_state
.. automethod:: culebra.trainer.aco.abc.SingleColACO._save_state
.. automethod:: culebra.trainer.aco.abc.SingleColACO._load_state
.. automethod:: culebra.trainer.aco.abc.SingleColACO._new_state
.. automethod:: culebra.trainer.aco.abc.SingleColACO._init_state
.. automethod:: culebra.trainer.aco.abc.SingleColACO._reset_state
.. automethod:: culebra.trainer.aco.abc.SingleColACO._init_internals
.. automethod:: culebra.trainer.aco.abc.SingleColACO._reset_internals
.. automethod:: culebra.trainer.aco.abc.SingleColACO._init_search
.. automethod:: culebra.trainer.aco.abc.SingleColACO._search
.. automethod:: culebra.trainer.aco.abc.SingleColACO._finish_search
.. automethod:: culebra.trainer.aco.abc.SingleColACO._start_iteration
.. automethod:: culebra.trainer.aco.abc.SingleColACO._preprocess_iteration
.. automethod:: culebra.trainer.aco.abc.SingleColACO._do_iteration
.. automethod:: culebra.trainer.aco.abc.SingleColACO._postprocess_iteration
.. automethod:: culebra.trainer.aco.abc.SingleColACO._finish_iteration
.. automethod:: culebra.trainer.aco.abc.SingleColACO._do_iteration_stats
.. automethod:: culebra.trainer.aco.abc.SingleColACO._default_termination_func
.. automethod:: culebra.trainer.aco.abc.SingleColACO._termination_criterion
.. automethod:: culebra.trainer.aco.abc.SingleColACO._init_representatives
.. automethod:: culebra.trainer.aco.abc.SingleColACO._calculate_choice_info
.. automethod:: culebra.trainer.aco.abc.SingleColACO._initial_choice
.. automethod:: culebra.trainer.aco.abc.SingleColACO._next_choice
.. automethod:: culebra.trainer.aco.abc.SingleColACO._feasible_neighborhood_probs
.. automethod:: culebra.trainer.aco.abc.SingleColACO._generate_ant
.. automethod:: culebra.trainer.aco.abc.SingleColACO._generate_col
.. automethod:: culebra.trainer.aco.abc.SingleColACO._deposit_pheromone
.. automethod:: culebra.trainer.aco.abc.SingleColACO._increase_pheromone
.. automethod:: culebra.trainer.aco.abc.SingleColACO._decrease_pheromone
.. automethod:: culebra.trainer.aco.abc.SingleColACO._update_pheromone

