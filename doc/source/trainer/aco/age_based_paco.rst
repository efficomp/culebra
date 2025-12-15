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

:class:`culebra.trainer.aco.QualityBasedPACO` class
===================================================

.. autoclass:: culebra.trainer.aco.QualityBasedPACO

Class attributes
----------------
.. autoattribute:: culebra.trainer.aco.QualityBasedPACO.objective_stats
.. autoattribute:: culebra.trainer.aco.QualityBasedPACO.stats_names

Class methods
-------------
.. automethod:: culebra.trainer.aco.QualityBasedPACO.load

Properties
----------
.. autoproperty:: culebra.trainer.aco.QualityBasedPACO.checkpoint_activation
.. autoproperty:: culebra.trainer.aco.QualityBasedPACO.checkpoint_filename
.. autoproperty:: culebra.trainer.aco.QualityBasedPACO.checkpoint_freq
.. autoproperty:: culebra.trainer.aco.QualityBasedPACO.choice_info
.. autoproperty:: culebra.trainer.aco.QualityBasedPACO.col
.. autoproperty:: culebra.trainer.aco.QualityBasedPACO.col_size
.. autoproperty:: culebra.trainer.aco.QualityBasedPACO.container
.. autoproperty:: culebra.trainer.aco.QualityBasedPACO.current_iter
.. autoproperty:: culebra.trainer.aco.QualityBasedPACO.custom_termination_func
.. autoproperty:: culebra.trainer.aco.QualityBasedPACO.exploitation_prob
.. autoproperty:: culebra.trainer.aco.QualityBasedPACO.fitness_function
.. autoproperty:: culebra.trainer.aco.QualityBasedPACO.heuristic
.. autoproperty:: culebra.trainer.aco.QualityBasedPACO.heuristic_influence
.. autoproperty:: culebra.trainer.aco.QualityBasedPACO.heuristic_shapes
.. autoproperty:: culebra.trainer.aco.QualityBasedPACO.index
.. autoproperty:: culebra.trainer.aco.QualityBasedPACO.initial_pheromone
.. autoproperty:: culebra.trainer.aco.QualityBasedPACO.logbook
.. autoproperty:: culebra.trainer.aco.QualityBasedPACO.max_num_iters
.. autoproperty:: culebra.trainer.aco.QualityBasedPACO.max_pheromone
.. autoproperty:: culebra.trainer.aco.QualityBasedPACO.num_evals
.. autoproperty:: culebra.trainer.aco.QualityBasedPACO.num_heuristic_matrices
.. autoproperty:: culebra.trainer.aco.QualityBasedPACO.num_pheromone_matrices
.. autoproperty:: culebra.trainer.aco.QualityBasedPACO.pheromone
.. autoproperty:: culebra.trainer.aco.QualityBasedPACO.pheromone_influence
.. autoproperty:: culebra.trainer.aco.QualityBasedPACO.pheromone_shapes
.. autoproperty:: culebra.trainer.aco.QualityBasedPACO.pop
.. autoproperty:: culebra.trainer.aco.QualityBasedPACO.pop_ingoing
.. autoproperty:: culebra.trainer.aco.QualityBasedPACO.pop_outgoing
.. autoproperty:: culebra.trainer.aco.QualityBasedPACO.pop_size
.. autoproperty:: culebra.trainer.aco.QualityBasedPACO.random_seed
.. autoproperty:: culebra.trainer.aco.QualityBasedPACO.representatives
.. autoproperty:: culebra.trainer.aco.QualityBasedPACO.runtime
.. autoproperty:: culebra.trainer.aco.QualityBasedPACO.solution_cls
.. autoproperty:: culebra.trainer.aco.QualityBasedPACO.species
.. autoproperty:: culebra.trainer.aco.QualityBasedPACO.verbosity

Private properties
------------------
.. autoproperty:: culebra.trainer.aco.QualityBasedPACO._default_checkpoint_activation
.. autoproperty:: culebra.trainer.aco.QualityBasedPACO._default_checkpoint_filename
.. autoproperty:: culebra.trainer.aco.QualityBasedPACO._default_checkpoint_freq
.. autoproperty:: culebra.trainer.aco.QualityBasedPACO._default_col_size
.. autoproperty:: culebra.trainer.aco.QualityBasedPACO._default_exploitation_prob
.. autoproperty:: culebra.trainer.aco.QualityBasedPACO._default_heuristic
.. autoproperty:: culebra.trainer.aco.QualityBasedPACO._default_heuristic_influence
.. autoproperty:: culebra.trainer.aco.QualityBasedPACO._default_index
.. autoproperty:: culebra.trainer.aco.QualityBasedPACO._default_max_num_iters
.. autoproperty:: culebra.trainer.aco.QualityBasedPACO._default_pheromone_influence
.. autoproperty:: culebra.trainer.aco.QualityBasedPACO._default_pop_size
.. autoproperty:: culebra.trainer.aco.QualityBasedPACO._default_verbosity

Methods
-------
.. automethod:: culebra.trainer.aco.QualityBasedPACO.best_representatives
.. automethod:: culebra.trainer.aco.QualityBasedPACO.best_solutions
.. automethod:: culebra.trainer.aco.QualityBasedPACO.dump
.. automethod:: culebra.trainer.aco.QualityBasedPACO.evaluate
.. automethod:: culebra.trainer.aco.QualityBasedPACO.reset
.. automethod:: culebra.trainer.aco.QualityBasedPACO.test
.. automethod:: culebra.trainer.aco.QualityBasedPACO.train

Private methods
---------------
.. automethod:: culebra.trainer.aco.QualityBasedPACO._ant_choice_info
.. automethod:: culebra.trainer.aco.QualityBasedPACO._calculate_choice_info
.. automethod:: culebra.trainer.aco.QualityBasedPACO._decrease_pheromone
.. automethod:: culebra.trainer.aco.QualityBasedPACO._default_termination_func
.. automethod:: culebra.trainer.aco.QualityBasedPACO._deposit_pheromone
.. automethod:: culebra.trainer.aco.QualityBasedPACO._do_iteration
.. automethod:: culebra.trainer.aco.QualityBasedPACO._do_iteration_stats
.. automethod:: culebra.trainer.aco.QualityBasedPACO._finish_iteration
.. automethod:: culebra.trainer.aco.QualityBasedPACO._finish_search
.. automethod:: culebra.trainer.aco.QualityBasedPACO._generate_ant
.. automethod:: culebra.trainer.aco.QualityBasedPACO._generate_col
.. automethod:: culebra.trainer.aco.QualityBasedPACO._get_state
.. automethod:: culebra.trainer.aco.QualityBasedPACO._increase_pheromone
.. automethod:: culebra.trainer.aco.QualityBasedPACO._init_internals
.. automethod:: culebra.trainer.aco.QualityBasedPACO._init_pheromone
.. automethod:: culebra.trainer.aco.QualityBasedPACO._init_representatives
.. automethod:: culebra.trainer.aco.QualityBasedPACO._init_search
.. automethod:: culebra.trainer.aco.QualityBasedPACO._init_state
.. automethod:: culebra.trainer.aco.QualityBasedPACO._load_state
.. automethod:: culebra.trainer.aco.QualityBasedPACO._new_state
.. automethod:: culebra.trainer.aco.QualityBasedPACO._next_choice
.. automethod:: culebra.trainer.aco.QualityBasedPACO._pheromone_amount
.. automethod:: culebra.trainer.aco.QualityBasedPACO._postprocess_iteration
.. automethod:: culebra.trainer.aco.QualityBasedPACO._preprocess_iteration
.. automethod:: culebra.trainer.aco.QualityBasedPACO._reset_internals
.. automethod:: culebra.trainer.aco.QualityBasedPACO._reset_state
.. automethod:: culebra.trainer.aco.QualityBasedPACO._save_state
.. automethod:: culebra.trainer.aco.QualityBasedPACO._search
.. automethod:: culebra.trainer.aco.QualityBasedPACO._set_cooperative_fitness
.. automethod:: culebra.trainer.aco.QualityBasedPACO._set_state
.. automethod:: culebra.trainer.aco.QualityBasedPACO._start_iteration
.. automethod:: culebra.trainer.aco.QualityBasedPACO._termination_criterion
.. automethod:: culebra.trainer.aco.QualityBasedPACO._update_pheromone
.. automethod:: culebra.trainer.aco.QualityBasedPACO._update_pop
