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

:py:class:`culebra.trainer.aco.abc.ElitistACO` class
====================================================

.. autoclass:: culebra.trainer.aco.abc.ElitistACO

Class attributes
----------------
.. autoattribute:: culebra.trainer.aco.abc.ElitistACO.stats_names
.. autoattribute:: culebra.trainer.aco.abc.ElitistACO.objective_stats

Properties
----------
.. autoproperty:: culebra.trainer.aco.abc.ElitistACO.num_pheromone_matrices
.. autoproperty:: culebra.trainer.aco.abc.ElitistACO.num_heuristic_matrices
.. autoproperty:: culebra.trainer.aco.abc.ElitistACO.solution_cls
.. autoproperty:: culebra.trainer.aco.abc.ElitistACO.species
.. autoproperty:: culebra.trainer.aco.abc.ElitistACO.fitness_function
.. autoproperty:: culebra.trainer.aco.abc.ElitistACO.initial_pheromone
.. autoproperty:: culebra.trainer.aco.abc.ElitistACO.heuristic
.. autoproperty:: culebra.trainer.aco.abc.ElitistACO.pheromone_influence
.. autoproperty:: culebra.trainer.aco.abc.ElitistACO.heuristic_influence
.. autoproperty:: culebra.trainer.aco.abc.ElitistACO.pheromone
.. autoproperty:: culebra.trainer.aco.abc.ElitistACO.choice_info
.. autoproperty:: culebra.trainer.aco.abc.ElitistACO.max_num_iters
.. autoproperty:: culebra.trainer.aco.abc.ElitistACO.current_iter
.. autoproperty:: culebra.trainer.aco.abc.ElitistACO.custom_termination_func
.. autoproperty:: culebra.trainer.aco.abc.ElitistACO.col_size
.. autoproperty:: culebra.trainer.aco.abc.ElitistACO.checkpoint_enable
.. autoproperty:: culebra.trainer.aco.abc.ElitistACO.checkpoint_freq
.. autoproperty:: culebra.trainer.aco.abc.ElitistACO.checkpoint_filename
.. autoproperty:: culebra.trainer.aco.abc.ElitistACO.verbose
.. autoproperty:: culebra.trainer.aco.abc.ElitistACO.random_seed
.. autoproperty:: culebra.trainer.aco.abc.ElitistACO.logbook
.. autoproperty:: culebra.trainer.aco.abc.ElitistACO.num_evals
.. autoproperty:: culebra.trainer.aco.abc.ElitistACO.runtime
.. autoproperty:: culebra.trainer.aco.abc.ElitistACO.index
.. autoproperty:: culebra.trainer.aco.abc.ElitistACO.container
.. autoproperty:: culebra.trainer.aco.abc.ElitistACO.representatives
.. autoproperty:: culebra.trainer.aco.abc.ElitistACO.col

Methods
-------
.. automethod:: culebra.trainer.aco.abc.ElitistACO.reset
.. automethod:: culebra.trainer.aco.abc.ElitistACO.evaluate
.. automethod:: culebra.trainer.aco.abc.ElitistACO.best_solutions
.. automethod:: culebra.trainer.aco.abc.ElitistACO.best_representatives
.. automethod:: culebra.trainer.aco.abc.ElitistACO.train
.. automethod:: culebra.trainer.aco.abc.ElitistACO.test

Private methods
---------------
.. automethod:: culebra.trainer.aco.abc.ElitistACO._get_state
.. automethod:: culebra.trainer.aco.abc.ElitistACO._set_state
.. automethod:: culebra.trainer.aco.abc.ElitistACO._save_state
.. automethod:: culebra.trainer.aco.abc.ElitistACO._load_state
.. automethod:: culebra.trainer.aco.abc.ElitistACO._new_state
.. automethod:: culebra.trainer.aco.abc.ElitistACO._init_state
.. automethod:: culebra.trainer.aco.abc.ElitistACO._reset_state
.. automethod:: culebra.trainer.aco.abc.ElitistACO._init_internals
.. automethod:: culebra.trainer.aco.abc.ElitistACO._reset_internals
.. automethod:: culebra.trainer.aco.abc.ElitistACO._init_search
.. automethod:: culebra.trainer.aco.abc.ElitistACO._search
.. automethod:: culebra.trainer.aco.abc.ElitistACO._finish_search
.. automethod:: culebra.trainer.aco.abc.ElitistACO._start_iteration
.. automethod:: culebra.trainer.aco.abc.ElitistACO._preprocess_iteration
.. automethod:: culebra.trainer.aco.abc.ElitistACO._do_iteration
.. automethod:: culebra.trainer.aco.abc.ElitistACO._postprocess_iteration
.. automethod:: culebra.trainer.aco.abc.ElitistACO._finish_iteration
.. automethod:: culebra.trainer.aco.abc.ElitistACO._do_iteration_stats
.. automethod:: culebra.trainer.aco.abc.ElitistACO._default_termination_func
.. automethod:: culebra.trainer.aco.abc.ElitistACO._termination_criterion
.. automethod:: culebra.trainer.aco.abc.ElitistACO._init_representatives
.. automethod:: culebra.trainer.aco.abc.ElitistACO._calculate_choice_info
.. automethod:: culebra.trainer.aco.abc.ElitistACO._initial_choice
.. automethod:: culebra.trainer.aco.abc.ElitistACO._next_choice
.. automethod:: culebra.trainer.aco.abc.ElitistACO._feasible_neighborhood_probs
.. automethod:: culebra.trainer.aco.abc.ElitistACO._generate_ant
.. automethod:: culebra.trainer.aco.abc.ElitistACO._generate_col
.. automethod:: culebra.trainer.aco.abc.ElitistACO._deposit_pheromone
.. automethod:: culebra.trainer.aco.abc.ElitistACO._increase_pheromone
.. automethod:: culebra.trainer.aco.abc.ElitistACO._decrease_pheromone
.. automethod:: culebra.trainer.aco.abc.ElitistACO._update_pheromone
.. automethod:: culebra.trainer.aco.abc.ElitistACO._update_elite
