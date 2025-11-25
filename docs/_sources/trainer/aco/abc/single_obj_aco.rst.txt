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

:class:`culebra.trainer.aco.abc.SingleObjACO` class
===================================================

.. autoclass:: culebra.trainer.aco.abc.SingleObjACO

Class attributes
----------------
.. autoattribute:: culebra.trainer.aco.abc.SingleObjACO.objective_stats
.. autoattribute:: culebra.trainer.aco.abc.SingleObjACO.stats_names

Class methods
-------------
.. automethod:: culebra.trainer.aco.abc.SingleObjACO.load

Properties
----------
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.checkpoint_enable
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.checkpoint_filename
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.checkpoint_freq
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.choice_info
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.col
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.col_size
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.container
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.current_iter
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.custom_termination_func
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.exploitation_prob
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.fitness_function
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.heuristic
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.heuristic_influence
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.heuristic_shapes
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.index
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.initial_pheromone
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.logbook
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.max_num_iters
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.num_evals
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.num_heuristic_matrices
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.num_pheromone_matrices
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.pheromone
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.pheromone_influence
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.pheromone_shapes
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.random_seed
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.representatives
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.runtime
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.solution_cls
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.species
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.verbose

Methods
-------
.. automethod:: culebra.trainer.aco.abc.SingleObjACO.best_representatives
.. automethod:: culebra.trainer.aco.abc.SingleObjACO.best_solutions
.. automethod:: culebra.trainer.aco.abc.SingleObjACO.dump
.. automethod:: culebra.trainer.aco.abc.SingleObjACO.evaluate
.. automethod:: culebra.trainer.aco.abc.SingleObjACO.reset
.. automethod:: culebra.trainer.aco.abc.SingleObjACO.test
.. automethod:: culebra.trainer.aco.abc.SingleObjACO.train

Private methods
---------------
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._ant_choice_info
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._calculate_choice_info
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._decrease_pheromone
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._default_termination_func
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._deposit_pheromone
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._do_iteration
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._do_iteration_stats
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._finish_iteration
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._finish_search
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._generate_ant
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._generate_col
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._get_state
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._increase_pheromone
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._init_internals
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._init_pheromone
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._init_representatives
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._init_search
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._init_state
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._load_state
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._new_state
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._next_choice
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._pheromone_amount
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._postprocess_iteration
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._preprocess_iteration
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._reset_internals
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._reset_state
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._save_state
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._search
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._set_cooperative_fitness
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._set_state
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._start_iteration
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._termination_criterion
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._update_pheromone
