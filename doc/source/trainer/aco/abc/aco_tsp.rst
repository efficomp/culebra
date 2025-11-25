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

:class:`culebra.trainer.aco.abc.ACOTSP` class
=============================================

.. autoclass:: culebra.trainer.aco.abc.ACOTSP

Class attributes
----------------
.. autoattribute:: culebra.trainer.aco.abc.ACOTSP.objective_stats
.. autoattribute:: culebra.trainer.aco.abc.ACOTSP.stats_names

Class methods
-------------
.. automethod:: culebra.trainer.aco.abc.ACOTSP.load

Properties
----------
.. autoproperty:: culebra.trainer.aco.abc.ACOTSP.checkpoint_enable
.. autoproperty:: culebra.trainer.aco.abc.ACOTSP.checkpoint_filename
.. autoproperty:: culebra.trainer.aco.abc.ACOTSP.checkpoint_freq
.. autoproperty:: culebra.trainer.aco.abc.ACOTSP.choice_info
.. autoproperty:: culebra.trainer.aco.abc.ACOTSP.col
.. autoproperty:: culebra.trainer.aco.abc.ACOTSP.col_size
.. autoproperty:: culebra.trainer.aco.abc.ACOTSP.container
.. autoproperty:: culebra.trainer.aco.abc.ACOTSP.current_iter
.. autoproperty:: culebra.trainer.aco.abc.ACOTSP.custom_termination_func
.. autoproperty:: culebra.trainer.aco.abc.ACOTSP.exploitation_prob
.. autoproperty:: culebra.trainer.aco.abc.ACOTSP.fitness_function
.. autoproperty:: culebra.trainer.aco.abc.ACOTSP.heuristic
.. autoproperty:: culebra.trainer.aco.abc.ACOTSP.heuristic_influence
.. autoproperty:: culebra.trainer.aco.abc.ACOTSP.heuristic_shapes
.. autoproperty:: culebra.trainer.aco.abc.ACOTSP.index
.. autoproperty:: culebra.trainer.aco.abc.ACOTSP.initial_pheromone
.. autoproperty:: culebra.trainer.aco.abc.ACOTSP.logbook
.. autoproperty:: culebra.trainer.aco.abc.ACOTSP.max_num_iters
.. autoproperty:: culebra.trainer.aco.abc.ACOTSP.num_evals
.. autoproperty:: culebra.trainer.aco.abc.ACOTSP.num_heuristic_matrices
.. autoproperty:: culebra.trainer.aco.abc.ACOTSP.num_pheromone_matrices
.. autoproperty:: culebra.trainer.aco.abc.ACOTSP.pheromone
.. autoproperty:: culebra.trainer.aco.abc.ACOTSP.pheromone_influence
.. autoproperty:: culebra.trainer.aco.abc.ACOTSP.pheromone_shapes
.. autoproperty:: culebra.trainer.aco.abc.ACOTSP.random_seed
.. autoproperty:: culebra.trainer.aco.abc.ACOTSP.representatives
.. autoproperty:: culebra.trainer.aco.abc.ACOTSP.runtime
.. autoproperty:: culebra.trainer.aco.abc.ACOTSP.solution_cls
.. autoproperty:: culebra.trainer.aco.abc.ACOTSP.species
.. autoproperty:: culebra.trainer.aco.abc.ACOTSP.verbose

Methods
-------
.. automethod:: culebra.trainer.aco.abc.ACOTSP.best_representatives
.. automethod:: culebra.trainer.aco.abc.ACOTSP.best_solutions
.. automethod:: culebra.trainer.aco.abc.ACOTSP.dump
.. automethod:: culebra.trainer.aco.abc.ACOTSP.evaluate
.. automethod:: culebra.trainer.aco.abc.ACOTSP.reset
.. automethod:: culebra.trainer.aco.abc.ACOTSP.test
.. automethod:: culebra.trainer.aco.abc.ACOTSP.train

Private methods
---------------
.. automethod:: culebra.trainer.aco.abc.ACOTSP._ant_choice_info
.. automethod:: culebra.trainer.aco.abc.ACOTSP._calculate_choice_info
.. automethod:: culebra.trainer.aco.abc.ACOTSP._decrease_pheromone
.. automethod:: culebra.trainer.aco.abc.ACOTSP._default_termination_func
.. automethod:: culebra.trainer.aco.abc.ACOTSP._deposit_pheromone
.. automethod:: culebra.trainer.aco.abc.ACOTSP._do_iteration
.. automethod:: culebra.trainer.aco.abc.ACOTSP._do_iteration_stats
.. automethod:: culebra.trainer.aco.abc.ACOTSP._finish_iteration
.. automethod:: culebra.trainer.aco.abc.ACOTSP._finish_search
.. automethod:: culebra.trainer.aco.abc.ACOTSP._generate_ant
.. automethod:: culebra.trainer.aco.abc.ACOTSP._generate_col
.. automethod:: culebra.trainer.aco.abc.ACOTSP._get_state
.. automethod:: culebra.trainer.aco.abc.ACOTSP._increase_pheromone
.. automethod:: culebra.trainer.aco.abc.ACOTSP._init_internals
.. automethod:: culebra.trainer.aco.abc.ACOTSP._init_pheromone
.. automethod:: culebra.trainer.aco.abc.ACOTSP._init_representatives
.. automethod:: culebra.trainer.aco.abc.ACOTSP._init_search
.. automethod:: culebra.trainer.aco.abc.ACOTSP._init_state
.. automethod:: culebra.trainer.aco.abc.ACOTSP._load_state
.. automethod:: culebra.trainer.aco.abc.ACOTSP._new_state
.. automethod:: culebra.trainer.aco.abc.ACOTSP._next_choice
.. automethod:: culebra.trainer.aco.abc.ACOTSP._pheromone_amount
.. automethod:: culebra.trainer.aco.abc.ACOTSP._postprocess_iteration
.. automethod:: culebra.trainer.aco.abc.ACOTSP._preprocess_iteration
.. automethod:: culebra.trainer.aco.abc.ACOTSP._reset_internals
.. automethod:: culebra.trainer.aco.abc.ACOTSP._reset_state
.. automethod:: culebra.trainer.aco.abc.ACOTSP._save_state
.. automethod:: culebra.trainer.aco.abc.ACOTSP._search
.. automethod:: culebra.trainer.aco.abc.ACOTSP._set_cooperative_fitness
.. automethod:: culebra.trainer.aco.abc.ACOTSP._set_state
.. automethod:: culebra.trainer.aco.abc.ACOTSP._start_iteration
.. automethod:: culebra.trainer.aco.abc.ACOTSP._termination_criterion
.. automethod:: culebra.trainer.aco.abc.ACOTSP._update_pheromone
