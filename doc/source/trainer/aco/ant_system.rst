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

:py:class:`culebra.trainer.aco.AntSystem` class
===============================================

.. autoclass:: culebra.trainer.aco.AntSystem

Class attributes
----------------

.. autoattribute:: culebra.trainer.aco.AntSystem.stats_names
.. autoattribute:: culebra.trainer.aco.AntSystem.objective_stats

Properties
----------

.. autoproperty:: culebra.trainer.aco.AntSystem.solution_cls
.. autoproperty:: culebra.trainer.aco.AntSystem.species
.. autoproperty:: culebra.trainer.aco.AntSystem.fitness_function
.. autoproperty:: culebra.trainer.aco.AntSystem.initial_pheromones
.. autoproperty:: culebra.trainer.aco.AntSystem.heuristics
.. autoproperty:: culebra.trainer.aco.AntSystem.pheromones
.. autoproperty:: culebra.trainer.aco.AntSystem.choice_info
.. autoproperty:: culebra.trainer.aco.AntSystem.pheromones_influence
.. autoproperty:: culebra.trainer.aco.AntSystem.heuristics_influence
.. autoproperty:: culebra.trainer.aco.AntSystem.pheromone_evaporation_rate
.. autoproperty:: culebra.trainer.aco.AntSystem.max_num_iters
.. autoproperty:: culebra.trainer.aco.AntSystem.current_iter
.. autoproperty:: culebra.trainer.aco.AntSystem.custom_termination_func
.. autoproperty:: culebra.trainer.aco.AntSystem.col_size
.. autoproperty:: culebra.trainer.aco.AntSystem.checkpoint_enable
.. autoproperty:: culebra.trainer.aco.AntSystem.checkpoint_freq
.. autoproperty:: culebra.trainer.aco.AntSystem.checkpoint_filename
.. autoproperty:: culebra.trainer.aco.AntSystem.verbose
.. autoproperty:: culebra.trainer.aco.AntSystem.random_seed
.. autoproperty:: culebra.trainer.aco.AntSystem.logbook
.. autoproperty:: culebra.trainer.aco.AntSystem.num_evals
.. autoproperty:: culebra.trainer.aco.AntSystem.runtime
.. autoproperty:: culebra.trainer.aco.AntSystem.index
.. autoproperty:: culebra.trainer.aco.AntSystem.container
.. autoproperty:: culebra.trainer.aco.AntSystem.representatives
.. autoproperty:: culebra.trainer.aco.AntSystem.col

Methods
-------

.. automethod:: culebra.trainer.aco.AntSystem.reset
.. automethod:: culebra.trainer.aco.AntSystem.evaluate
.. automethod:: culebra.trainer.aco.AntSystem.best_solutions
.. automethod:: culebra.trainer.aco.AntSystem.best_representatives
.. automethod:: culebra.trainer.aco.AntSystem.train
.. automethod:: culebra.trainer.aco.AntSystem.test

Private methods
---------------

.. automethod:: culebra.trainer.aco.AntSystem._get_state
.. automethod:: culebra.trainer.aco.AntSystem._set_state
.. automethod:: culebra.trainer.aco.AntSystem._save_state
.. automethod:: culebra.trainer.aco.AntSystem._load_state
.. automethod:: culebra.trainer.aco.AntSystem._new_state
.. automethod:: culebra.trainer.aco.AntSystem._init_state
.. automethod:: culebra.trainer.aco.AntSystem._reset_state
.. automethod:: culebra.trainer.aco.AntSystem._init_internals
.. automethod:: culebra.trainer.aco.AntSystem._reset_internals
.. automethod:: culebra.trainer.aco.AntSystem._init_search
.. automethod:: culebra.trainer.aco.AntSystem._search
.. automethod:: culebra.trainer.aco.AntSystem._finish_search
.. automethod:: culebra.trainer.aco.AntSystem._start_iteration
.. automethod:: culebra.trainer.aco.AntSystem._preprocess_iteration
.. automethod:: culebra.trainer.aco.AntSystem._do_iteration
.. automethod:: culebra.trainer.aco.AntSystem._postprocess_iteration
.. automethod:: culebra.trainer.aco.AntSystem._finish_iteration
.. automethod:: culebra.trainer.aco.AntSystem._do_iteration_stats
.. automethod:: culebra.trainer.aco.AntSystem._default_termination_func
.. automethod:: culebra.trainer.aco.AntSystem._termination_criterion
.. automethod:: culebra.trainer.aco.AntSystem._init_representatives
.. automethod:: culebra.trainer.aco.AntSystem._calculate_choice_info
.. automethod:: culebra.trainer.aco.AntSystem._initial_choice
.. automethod:: culebra.trainer.aco.AntSystem._next_choice
.. automethod:: culebra.trainer.aco.AntSystem._feasible_neighborhood_probs
.. automethod:: culebra.trainer.aco.AntSystem._generate_ant
.. automethod:: culebra.trainer.aco.AntSystem._generate_col
.. automethod:: culebra.trainer.aco.AntSystem._deposit_pheromones
.. automethod:: culebra.trainer.aco.AntSystem._increase_pheromones
.. automethod:: culebra.trainer.aco.AntSystem._decrease_pheromones
.. automethod:: culebra.trainer.aco.AntSystem._update_pheromones
