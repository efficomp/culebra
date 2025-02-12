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

:py:class:`culebra.trainer.aco.abc.MaxPheromonePACO` class
==========================================================

.. autoclass:: culebra.trainer.aco.abc.MaxPheromonePACO

Class attributes
----------------
.. autoattribute:: culebra.trainer.aco.abc.MaxPheromonePACO.stats_names
.. autoattribute:: culebra.trainer.aco.abc.MaxPheromonePACO.objective_stats

Class methods
-------------
.. automethod:: culebra.trainer.aco.abc.MaxPheromonePACO.load_pickle

Properties
----------
.. autoproperty:: culebra.trainer.aco.abc.MaxPheromonePACO.num_pheromone_matrices
.. autoproperty:: culebra.trainer.aco.abc.MaxPheromonePACO.num_heuristic_matrices
.. autoproperty:: culebra.trainer.aco.abc.MaxPheromonePACO.solution_cls
.. autoproperty:: culebra.trainer.aco.abc.MaxPheromonePACO.species
.. autoproperty:: culebra.trainer.aco.abc.MaxPheromonePACO.fitness_function
.. autoproperty:: culebra.trainer.aco.abc.MaxPheromonePACO.initial_pheromone
.. autoproperty:: culebra.trainer.aco.abc.MaxPheromonePACO.max_pheromone
.. autoproperty:: culebra.trainer.aco.abc.MaxPheromonePACO.heuristic
.. autoproperty:: culebra.trainer.aco.abc.MaxPheromonePACO.pheromone_influence
.. autoproperty:: culebra.trainer.aco.abc.MaxPheromonePACO.heuristic_influence
.. autoproperty:: culebra.trainer.aco.abc.MaxPheromonePACO.pheromone
.. autoproperty:: culebra.trainer.aco.abc.MaxPheromonePACO.choice_info
.. autoproperty:: culebra.trainer.aco.abc.MaxPheromonePACO.max_num_iters
.. autoproperty:: culebra.trainer.aco.abc.MaxPheromonePACO.current_iter
.. autoproperty:: culebra.trainer.aco.abc.MaxPheromonePACO.custom_termination_func
.. autoproperty:: culebra.trainer.aco.abc.MaxPheromonePACO.col_size
.. autoproperty:: culebra.trainer.aco.abc.MaxPheromonePACO.pop_size
.. autoproperty:: culebra.trainer.aco.abc.MaxPheromonePACO.checkpoint_enable
.. autoproperty:: culebra.trainer.aco.abc.MaxPheromonePACO.checkpoint_freq
.. autoproperty:: culebra.trainer.aco.abc.MaxPheromonePACO.checkpoint_filename
.. autoproperty:: culebra.trainer.aco.abc.MaxPheromonePACO.verbose
.. autoproperty:: culebra.trainer.aco.abc.MaxPheromonePACO.random_seed
.. autoproperty:: culebra.trainer.aco.abc.MaxPheromonePACO.logbook
.. autoproperty:: culebra.trainer.aco.abc.MaxPheromonePACO.num_evals
.. autoproperty:: culebra.trainer.aco.abc.MaxPheromonePACO.runtime
.. autoproperty:: culebra.trainer.aco.abc.MaxPheromonePACO.index
.. autoproperty:: culebra.trainer.aco.abc.MaxPheromonePACO.container
.. autoproperty:: culebra.trainer.aco.abc.MaxPheromonePACO.representatives
.. autoproperty:: culebra.trainer.aco.abc.MaxPheromonePACO.col
.. autoproperty:: culebra.trainer.aco.abc.MaxPheromonePACO.pop

Methods
-------
.. automethod:: culebra.trainer.aco.abc.MaxPheromonePACO.save_pickle
.. automethod:: culebra.trainer.aco.abc.MaxPheromonePACO.reset
.. automethod:: culebra.trainer.aco.abc.MaxPheromonePACO.evaluate
.. automethod:: culebra.trainer.aco.abc.MaxPheromonePACO.best_solutions
.. automethod:: culebra.trainer.aco.abc.MaxPheromonePACO.best_representatives
.. automethod:: culebra.trainer.aco.abc.MaxPheromonePACO.train
.. automethod:: culebra.trainer.aco.abc.MaxPheromonePACO.test

Private methods
---------------
.. automethod:: culebra.trainer.aco.abc.MaxPheromonePACO._get_state
.. automethod:: culebra.trainer.aco.abc.MaxPheromonePACO._set_state
.. automethod:: culebra.trainer.aco.abc.MaxPheromonePACO._save_state
.. automethod:: culebra.trainer.aco.abc.MaxPheromonePACO._load_state
.. automethod:: culebra.trainer.aco.abc.MaxPheromonePACO._new_state
.. automethod:: culebra.trainer.aco.abc.MaxPheromonePACO._init_state
.. automethod:: culebra.trainer.aco.abc.MaxPheromonePACO._reset_state
.. automethod:: culebra.trainer.aco.abc.MaxPheromonePACO._init_internals
.. automethod:: culebra.trainer.aco.abc.MaxPheromonePACO._reset_internals
.. automethod:: culebra.trainer.aco.abc.MaxPheromonePACO._init_search
.. automethod:: culebra.trainer.aco.abc.MaxPheromonePACO._search
.. automethod:: culebra.trainer.aco.abc.MaxPheromonePACO._finish_search
.. automethod:: culebra.trainer.aco.abc.MaxPheromonePACO._start_iteration
.. automethod:: culebra.trainer.aco.abc.MaxPheromonePACO._preprocess_iteration
.. automethod:: culebra.trainer.aco.abc.MaxPheromonePACO._do_iteration
.. automethod:: culebra.trainer.aco.abc.MaxPheromonePACO._postprocess_iteration
.. automethod:: culebra.trainer.aco.abc.MaxPheromonePACO._finish_iteration
.. automethod:: culebra.trainer.aco.abc.MaxPheromonePACO._do_iteration_stats
.. automethod:: culebra.trainer.aco.abc.MaxPheromonePACO._default_termination_func
.. automethod:: culebra.trainer.aco.abc.MaxPheromonePACO._termination_criterion
.. automethod:: culebra.trainer.aco.abc.MaxPheromonePACO._init_representatives
.. automethod:: culebra.trainer.aco.abc.MaxPheromonePACO._calculate_choice_info
.. automethod:: culebra.trainer.aco.abc.MaxPheromonePACO._initial_choice
.. automethod:: culebra.trainer.aco.abc.MaxPheromonePACO._next_choice
.. automethod:: culebra.trainer.aco.abc.MaxPheromonePACO._feasible_neighborhood_probs
.. automethod:: culebra.trainer.aco.abc.MaxPheromonePACO._generate_ant
.. automethod:: culebra.trainer.aco.abc.MaxPheromonePACO._generate_col
.. automethod:: culebra.trainer.aco.abc.MaxPheromonePACO._init_pheromone
.. automethod:: culebra.trainer.aco.abc.MaxPheromonePACO._deposit_pheromone
.. automethod:: culebra.trainer.aco.abc.MaxPheromonePACO._increase_pheromone
.. automethod:: culebra.trainer.aco.abc.MaxPheromonePACO._decrease_pheromone
.. automethod:: culebra.trainer.aco.abc.MaxPheromonePACO._update_pheromone
.. automethod:: culebra.trainer.aco.abc.MaxPheromonePACO._update_pop

