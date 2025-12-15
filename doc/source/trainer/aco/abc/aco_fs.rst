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

:class:`culebra.trainer.aco.abc.ACOFS` class
============================================

.. autoclass:: culebra.trainer.aco.abc.ACOFS

Class attributes
----------------
.. autoattribute:: culebra.trainer.aco.abc.ACOFS.objective_stats
.. autoattribute:: culebra.trainer.aco.abc.ACOFS.stats_names

Class methods
-------------
.. automethod:: culebra.trainer.aco.abc.ACOFS.load

Properties
----------
.. autoproperty:: culebra.trainer.aco.abc.ACOFS.checkpoint_activation
.. autoproperty:: culebra.trainer.aco.abc.ACOFS.checkpoint_filename
.. autoproperty:: culebra.trainer.aco.abc.ACOFS.checkpoint_freq
.. autoproperty:: culebra.trainer.aco.abc.ACOFS.choice_info
.. autoproperty:: culebra.trainer.aco.abc.ACOFS.col
.. autoproperty:: culebra.trainer.aco.abc.ACOFS.col_size
.. autoproperty:: culebra.trainer.aco.abc.ACOFS.container
.. autoproperty:: culebra.trainer.aco.abc.ACOFS.current_iter
.. autoproperty:: culebra.trainer.aco.abc.ACOFS.custom_termination_func
.. autoproperty:: culebra.trainer.aco.abc.ACOFS.discard_prob
.. autoproperty:: culebra.trainer.aco.abc.ACOFS.exploitation_prob
.. autoproperty:: culebra.trainer.aco.abc.ACOFS.fitness_function
.. autoproperty:: culebra.trainer.aco.abc.ACOFS.heuristic
.. autoproperty:: culebra.trainer.aco.abc.ACOFS.heuristic_influence
.. autoproperty:: culebra.trainer.aco.abc.ACOFS.heuristic_shapes
.. autoproperty:: culebra.trainer.aco.abc.ACOFS.index
.. autoproperty:: culebra.trainer.aco.abc.ACOFS.initial_pheromone
.. autoproperty:: culebra.trainer.aco.abc.ACOFS.logbook
.. autoproperty:: culebra.trainer.aco.abc.ACOFS.max_num_iters
.. autoproperty:: culebra.trainer.aco.abc.ACOFS.num_evals
.. autoproperty:: culebra.trainer.aco.abc.ACOFS.num_heuristic_matrices
.. autoproperty:: culebra.trainer.aco.abc.ACOFS.num_pheromone_matrices
.. autoproperty:: culebra.trainer.aco.abc.ACOFS.pheromone
.. autoproperty:: culebra.trainer.aco.abc.ACOFS.pheromone_influence
.. autoproperty:: culebra.trainer.aco.abc.ACOFS.pheromone_shapes
.. autoproperty:: culebra.trainer.aco.abc.ACOFS.random_seed
.. autoproperty:: culebra.trainer.aco.abc.ACOFS.representatives
.. autoproperty:: culebra.trainer.aco.abc.ACOFS.runtime
.. autoproperty:: culebra.trainer.aco.abc.ACOFS.solution_cls
.. autoproperty:: culebra.trainer.aco.abc.ACOFS.species
.. autoproperty:: culebra.trainer.aco.abc.ACOFS.verbosity

Private properties
------------------
.. autoproperty:: culebra.trainer.aco.abc.ACOFS._default_checkpoint_activation
.. autoproperty:: culebra.trainer.aco.abc.ACOFS._default_checkpoint_filename
.. autoproperty:: culebra.trainer.aco.abc.ACOFS._default_checkpoint_freq
.. autoproperty:: culebra.trainer.aco.abc.ACOFS._default_col_size
.. autoproperty:: culebra.trainer.aco.abc.ACOFS._default_discard_prob
.. autoproperty:: culebra.trainer.aco.abc.ACOFS._default_exploitation_prob
.. autoproperty:: culebra.trainer.aco.abc.ACOFS._default_heuristic
.. autoproperty:: culebra.trainer.aco.abc.ACOFS._default_heuristic_influence
.. autoproperty:: culebra.trainer.aco.abc.ACOFS._default_index
.. autoproperty:: culebra.trainer.aco.abc.ACOFS._default_initial_pheromone
.. autoproperty:: culebra.trainer.aco.abc.ACOFS._default_max_num_iters
.. autoproperty:: culebra.trainer.aco.abc.ACOFS._default_pheromone_influence
.. autoproperty:: culebra.trainer.aco.abc.ACOFS._default_verbosity

Methods
-------
.. automethod:: culebra.trainer.aco.abc.ACOFS.best_representatives
.. automethod:: culebra.trainer.aco.abc.ACOFS.best_solutions
.. automethod:: culebra.trainer.aco.abc.ACOFS.dump
.. automethod:: culebra.trainer.aco.abc.ACOFS.evaluate
.. automethod:: culebra.trainer.aco.abc.ACOFS.reset
.. automethod:: culebra.trainer.aco.abc.ACOFS.test
.. automethod:: culebra.trainer.aco.abc.ACOFS.train

Private methods
---------------
.. automethod:: culebra.trainer.aco.abc.ACOFS._ant_choice_info
.. automethod:: culebra.trainer.aco.abc.ACOFS._calculate_choice_info
.. automethod:: culebra.trainer.aco.abc.ACOFS._default_termination_func
.. automethod:: culebra.trainer.aco.abc.ACOFS._deposit_pheromone
.. automethod:: culebra.trainer.aco.abc.ACOFS._do_iteration
.. automethod:: culebra.trainer.aco.abc.ACOFS._do_iteration_stats
.. automethod:: culebra.trainer.aco.abc.ACOFS._finish_iteration
.. automethod:: culebra.trainer.aco.abc.ACOFS._finish_search
.. automethod:: culebra.trainer.aco.abc.ACOFS._generate_ant
.. automethod:: culebra.trainer.aco.abc.ACOFS._generate_col
.. automethod:: culebra.trainer.aco.abc.ACOFS._get_state
.. automethod:: culebra.trainer.aco.abc.ACOFS._init_internals
.. automethod:: culebra.trainer.aco.abc.ACOFS._init_pheromone
.. automethod:: culebra.trainer.aco.abc.ACOFS._init_representatives
.. automethod:: culebra.trainer.aco.abc.ACOFS._init_search
.. automethod:: culebra.trainer.aco.abc.ACOFS._init_state
.. automethod:: culebra.trainer.aco.abc.ACOFS._load_state
.. automethod:: culebra.trainer.aco.abc.ACOFS._new_state
.. automethod:: culebra.trainer.aco.abc.ACOFS._next_choice
.. automethod:: culebra.trainer.aco.abc.ACOFS._pheromone_amount
.. automethod:: culebra.trainer.aco.abc.ACOFS._postprocess_iteration
.. automethod:: culebra.trainer.aco.abc.ACOFS._preprocess_iteration
.. automethod:: culebra.trainer.aco.abc.ACOFS._reset_internals
.. automethod:: culebra.trainer.aco.abc.ACOFS._reset_state
.. automethod:: culebra.trainer.aco.abc.ACOFS._save_state
.. automethod:: culebra.trainer.aco.abc.ACOFS._search
.. automethod:: culebra.trainer.aco.abc.ACOFS._set_cooperative_fitness
.. automethod:: culebra.trainer.aco.abc.ACOFS._set_state
.. automethod:: culebra.trainer.aco.abc.ACOFS._start_iteration
.. automethod:: culebra.trainer.aco.abc.ACOFS._termination_criterion
.. automethod:: culebra.trainer.aco.abc.ACOFS._update_pheromone
