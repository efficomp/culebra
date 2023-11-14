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

:py:class:`culebra.trainer.aco.abc.QualityBasedPACO` class
==========================================================

.. autoclass:: culebra.trainer.aco.abc.QualityBasedPACO

Class attributes
----------------
.. autoattribute:: culebra.trainer.aco.abc.QualityBasedPACO.stats_names
.. autoattribute:: culebra.trainer.aco.abc.QualityBasedPACO.objective_stats

Properties
----------
.. autoproperty:: culebra.trainer.aco.abc.QualityBasedPACO.solution_cls
.. autoproperty:: culebra.trainer.aco.abc.QualityBasedPACO.species
.. autoproperty:: culebra.trainer.aco.abc.QualityBasedPACO.fitness_function
.. autoproperty:: culebra.trainer.aco.abc.QualityBasedPACO.initial_pheromones
.. autoproperty:: culebra.trainer.aco.abc.QualityBasedPACO.max_pheromones
.. autoproperty:: culebra.trainer.aco.abc.QualityBasedPACO.heuristics
.. autoproperty:: culebra.trainer.aco.abc.QualityBasedPACO.pheromones_influence
.. autoproperty:: culebra.trainer.aco.abc.QualityBasedPACO.heuristics_influence
.. autoproperty:: culebra.trainer.aco.abc.QualityBasedPACO.pheromones
.. autoproperty:: culebra.trainer.aco.abc.QualityBasedPACO.choice_info
.. autoproperty:: culebra.trainer.aco.abc.QualityBasedPACO.max_num_iters
.. autoproperty:: culebra.trainer.aco.abc.QualityBasedPACO.current_iter
.. autoproperty:: culebra.trainer.aco.abc.QualityBasedPACO.custom_termination_func
.. autoproperty:: culebra.trainer.aco.abc.QualityBasedPACO.col_size
.. autoproperty:: culebra.trainer.aco.abc.QualityBasedPACO.pop_size
.. autoproperty:: culebra.trainer.aco.abc.QualityBasedPACO.checkpoint_enable
.. autoproperty:: culebra.trainer.aco.abc.QualityBasedPACO.checkpoint_freq
.. autoproperty:: culebra.trainer.aco.abc.QualityBasedPACO.checkpoint_filename
.. autoproperty:: culebra.trainer.aco.abc.QualityBasedPACO.verbose
.. autoproperty:: culebra.trainer.aco.abc.QualityBasedPACO.random_seed
.. autoproperty:: culebra.trainer.aco.abc.QualityBasedPACO.logbook
.. autoproperty:: culebra.trainer.aco.abc.QualityBasedPACO.num_evals
.. autoproperty:: culebra.trainer.aco.abc.QualityBasedPACO.runtime
.. autoproperty:: culebra.trainer.aco.abc.QualityBasedPACO.index
.. autoproperty:: culebra.trainer.aco.abc.QualityBasedPACO.container
.. autoproperty:: culebra.trainer.aco.abc.QualityBasedPACO.representatives
.. autoproperty:: culebra.trainer.aco.abc.QualityBasedPACO.col
.. autoproperty:: culebra.trainer.aco.abc.QualityBasedPACO.pop

Methods
-------
.. automethod:: culebra.trainer.aco.abc.QualityBasedPACO.reset
.. automethod:: culebra.trainer.aco.abc.QualityBasedPACO.evaluate
.. automethod:: culebra.trainer.aco.abc.QualityBasedPACO.best_solutions
.. automethod:: culebra.trainer.aco.abc.QualityBasedPACO.best_representatives
.. automethod:: culebra.trainer.aco.abc.QualityBasedPACO.train
.. automethod:: culebra.trainer.aco.abc.QualityBasedPACO.test

Private methods
---------------
.. automethod:: culebra.trainer.aco.abc.QualityBasedPACO._get_state
.. automethod:: culebra.trainer.aco.abc.QualityBasedPACO._set_state
.. automethod:: culebra.trainer.aco.abc.QualityBasedPACO._save_state
.. automethod:: culebra.trainer.aco.abc.QualityBasedPACO._load_state
.. automethod:: culebra.trainer.aco.abc.QualityBasedPACO._new_state
.. automethod:: culebra.trainer.aco.abc.QualityBasedPACO._init_state
.. automethod:: culebra.trainer.aco.abc.QualityBasedPACO._reset_state
.. automethod:: culebra.trainer.aco.abc.QualityBasedPACO._init_internals
.. automethod:: culebra.trainer.aco.abc.QualityBasedPACO._reset_internals
.. automethod:: culebra.trainer.aco.abc.QualityBasedPACO._init_search
.. automethod:: culebra.trainer.aco.abc.QualityBasedPACO._search
.. automethod:: culebra.trainer.aco.abc.QualityBasedPACO._finish_search
.. automethod:: culebra.trainer.aco.abc.QualityBasedPACO._start_iteration
.. automethod:: culebra.trainer.aco.abc.QualityBasedPACO._preprocess_iteration
.. automethod:: culebra.trainer.aco.abc.QualityBasedPACO._do_iteration
.. automethod:: culebra.trainer.aco.abc.QualityBasedPACO._postprocess_iteration
.. automethod:: culebra.trainer.aco.abc.QualityBasedPACO._finish_iteration
.. automethod:: culebra.trainer.aco.abc.QualityBasedPACO._do_iteration_stats
.. automethod:: culebra.trainer.aco.abc.QualityBasedPACO._default_termination_func
.. automethod:: culebra.trainer.aco.abc.QualityBasedPACO._termination_criterion
.. automethod:: culebra.trainer.aco.abc.QualityBasedPACO._init_representatives
.. automethod:: culebra.trainer.aco.abc.QualityBasedPACO._calculate_choice_info
.. automethod:: culebra.trainer.aco.abc.QualityBasedPACO._initial_choice
.. automethod:: culebra.trainer.aco.abc.QualityBasedPACO._next_choice
.. automethod:: culebra.trainer.aco.abc.QualityBasedPACO._feasible_neighborhood_probs
.. automethod:: culebra.trainer.aco.abc.QualityBasedPACO._generate_ant
.. automethod:: culebra.trainer.aco.abc.QualityBasedPACO._generate_col
.. automethod:: culebra.trainer.aco.abc.QualityBasedPACO._deposit_pheromones
.. automethod:: culebra.trainer.aco.abc.QualityBasedPACO._increase_pheromones
.. automethod:: culebra.trainer.aco.abc.QualityBasedPACO._decrease_pheromones
.. automethod:: culebra.trainer.aco.abc.QualityBasedPACO._update_pheromones
.. automethod:: culebra.trainer.aco.abc.QualityBasedPACO._update_pop

