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

:py:class:`culebra.trainer.aco.SingleObjAgeBasedPACO` class
===========================================================

.. autoclass:: culebra.trainer.aco.SingleObjAgeBasedPACO

Class attributes
----------------
.. autoattribute:: culebra.trainer.aco.SingleObjAgeBasedPACO.stats_names
.. autoattribute:: culebra.trainer.aco.SingleObjAgeBasedPACO.objective_stats

Properties
----------
.. autoproperty:: culebra.trainer.aco.SingleObjAgeBasedPACO.solution_cls
.. autoproperty:: culebra.trainer.aco.SingleObjAgeBasedPACO.species
.. autoproperty:: culebra.trainer.aco.SingleObjAgeBasedPACO.fitness_function
.. autoproperty:: culebra.trainer.aco.SingleObjAgeBasedPACO.initial_pheromones
.. autoproperty:: culebra.trainer.aco.SingleObjAgeBasedPACO.max_pheromones
.. autoproperty:: culebra.trainer.aco.SingleObjAgeBasedPACO.heuristics
.. autoproperty:: culebra.trainer.aco.SingleObjAgeBasedPACO.pheromones_influence
.. autoproperty:: culebra.trainer.aco.SingleObjAgeBasedPACO.heuristics_influence
.. autoproperty:: culebra.trainer.aco.SingleObjAgeBasedPACO.pheromones
.. autoproperty:: culebra.trainer.aco.SingleObjAgeBasedPACO.choice_info
.. autoproperty:: culebra.trainer.aco.SingleObjAgeBasedPACO.max_num_iters
.. autoproperty:: culebra.trainer.aco.SingleObjAgeBasedPACO.current_iter
.. autoproperty:: culebra.trainer.aco.SingleObjAgeBasedPACO.custom_termination_func
.. autoproperty:: culebra.trainer.aco.SingleObjAgeBasedPACO.col_size
.. autoproperty:: culebra.trainer.aco.SingleObjAgeBasedPACO.pop_size
.. autoproperty:: culebra.trainer.aco.SingleObjAgeBasedPACO.checkpoint_enable
.. autoproperty:: culebra.trainer.aco.SingleObjAgeBasedPACO.checkpoint_freq
.. autoproperty:: culebra.trainer.aco.SingleObjAgeBasedPACO.checkpoint_filename
.. autoproperty:: culebra.trainer.aco.SingleObjAgeBasedPACO.verbose
.. autoproperty:: culebra.trainer.aco.SingleObjAgeBasedPACO.random_seed
.. autoproperty:: culebra.trainer.aco.SingleObjAgeBasedPACO.logbook
.. autoproperty:: culebra.trainer.aco.SingleObjAgeBasedPACO.num_evals
.. autoproperty:: culebra.trainer.aco.SingleObjAgeBasedPACO.runtime
.. autoproperty:: culebra.trainer.aco.SingleObjAgeBasedPACO.index
.. autoproperty:: culebra.trainer.aco.SingleObjAgeBasedPACO.container
.. autoproperty:: culebra.trainer.aco.SingleObjAgeBasedPACO.representatives
.. autoproperty:: culebra.trainer.aco.SingleObjAgeBasedPACO.col
.. autoproperty:: culebra.trainer.aco.SingleObjAgeBasedPACO.pop

Methods
-------
.. automethod:: culebra.trainer.aco.SingleObjAgeBasedPACO.reset
.. automethod:: culebra.trainer.aco.SingleObjAgeBasedPACO.evaluate
.. automethod:: culebra.trainer.aco.SingleObjAgeBasedPACO.best_solutions
.. automethod:: culebra.trainer.aco.SingleObjAgeBasedPACO.best_representatives
.. automethod:: culebra.trainer.aco.SingleObjAgeBasedPACO.train
.. automethod:: culebra.trainer.aco.SingleObjAgeBasedPACO.test

Private methods
---------------
.. automethod:: culebra.trainer.aco.SingleObjAgeBasedPACO._get_state
.. automethod:: culebra.trainer.aco.SingleObjAgeBasedPACO._set_state
.. automethod:: culebra.trainer.aco.SingleObjAgeBasedPACO._save_state
.. automethod:: culebra.trainer.aco.SingleObjAgeBasedPACO._load_state
.. automethod:: culebra.trainer.aco.SingleObjAgeBasedPACO._new_state
.. automethod:: culebra.trainer.aco.SingleObjAgeBasedPACO._init_state
.. automethod:: culebra.trainer.aco.SingleObjAgeBasedPACO._reset_state
.. automethod:: culebra.trainer.aco.SingleObjAgeBasedPACO._init_internals
.. automethod:: culebra.trainer.aco.SingleObjAgeBasedPACO._reset_internals
.. automethod:: culebra.trainer.aco.SingleObjAgeBasedPACO._init_search
.. automethod:: culebra.trainer.aco.SingleObjAgeBasedPACO._search
.. automethod:: culebra.trainer.aco.SingleObjAgeBasedPACO._finish_search
.. automethod:: culebra.trainer.aco.SingleObjAgeBasedPACO._start_iteration
.. automethod:: culebra.trainer.aco.SingleObjAgeBasedPACO._preprocess_iteration
.. automethod:: culebra.trainer.aco.SingleObjAgeBasedPACO._do_iteration
.. automethod:: culebra.trainer.aco.SingleObjAgeBasedPACO._postprocess_iteration
.. automethod:: culebra.trainer.aco.SingleObjAgeBasedPACO._finish_iteration
.. automethod:: culebra.trainer.aco.SingleObjAgeBasedPACO._do_iteration_stats
.. automethod:: culebra.trainer.aco.SingleObjAgeBasedPACO._default_termination_func
.. automethod:: culebra.trainer.aco.SingleObjAgeBasedPACO._termination_criterion
.. automethod:: culebra.trainer.aco.SingleObjAgeBasedPACO._init_representatives
.. automethod:: culebra.trainer.aco.SingleObjAgeBasedPACO._calculate_choice_info
.. automethod:: culebra.trainer.aco.SingleObjAgeBasedPACO._initial_choice
.. automethod:: culebra.trainer.aco.SingleObjAgeBasedPACO._next_choice
.. automethod:: culebra.trainer.aco.SingleObjAgeBasedPACO._feasible_neighborhood_probs
.. automethod:: culebra.trainer.aco.SingleObjAgeBasedPACO._generate_ant
.. automethod:: culebra.trainer.aco.SingleObjAgeBasedPACO._generate_col
.. automethod:: culebra.trainer.aco.SingleObjAgeBasedPACO._deposit_pheromones
.. automethod:: culebra.trainer.aco.SingleObjAgeBasedPACO._increase_pheromones
.. automethod:: culebra.trainer.aco.SingleObjAgeBasedPACO._decrease_pheromones
.. automethod:: culebra.trainer.aco.SingleObjAgeBasedPACO._update_pheromones
.. automethod:: culebra.trainer.aco.SingleObjAgeBasedPACO._update_pop

