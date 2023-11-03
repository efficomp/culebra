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
    If not, see <http://www.gnu.org/licenses/>.

   This work was supported by project PGC2018-098813-B-C31 (Spanish "Ministerio
   de Ciencia, Innovación y Universidades"), and by the European Regional
   Development Fund (ERDF).

:py:class:`culebra.trainer.aco.ElitistAntSystem` class
======================================================

.. autoclass:: culebra.trainer.aco.ElitistAntSystem

Class attributes
----------------

.. autoattribute:: culebra.trainer.aco.ElitistAntSystem.stats_names
.. autoattribute:: culebra.trainer.aco.ElitistAntSystem.objective_stats

Properties
----------

.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.solution_cls
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.species
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.fitness_function
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.initial_pheromones
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.heuristics
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.pheromones_influence
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.heuristics_influence
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.pheromone_evaporation_rate
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.elite_weight
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.pheromones
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.choice_info
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.max_num_iters
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.current_iter
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.custom_termination_func
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.col_size
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.checkpoint_enable
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.checkpoint_freq
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.checkpoint_filename
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.verbose
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.random_seed
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.logbook
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.num_evals
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.runtime
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.index
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.container
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.representatives
.. autoproperty:: culebra.trainer.aco.ElitistAntSystem.col


Private properties
------------------

.. autoproperty:: culebra.trainer.aco.ElitistAntSystem._state

Methods
-------

.. automethod:: culebra.trainer.aco.ElitistAntSystem.reset
.. automethod:: culebra.trainer.aco.ElitistAntSystem.evaluate
.. automethod:: culebra.trainer.aco.ElitistAntSystem.best_solutions
.. automethod:: culebra.trainer.aco.ElitistAntSystem.best_representatives
.. automethod:: culebra.trainer.aco.ElitistAntSystem.train
.. automethod:: culebra.trainer.aco.ElitistAntSystem.test

Private methods
---------------

.. automethod:: culebra.trainer.aco.ElitistAntSystem._save_state
.. automethod:: culebra.trainer.aco.ElitistAntSystem._load_state
.. automethod:: culebra.trainer.aco.ElitistAntSystem._new_state
.. automethod:: culebra.trainer.aco.ElitistAntSystem._init_state
.. automethod:: culebra.trainer.aco.ElitistAntSystem._reset_state
.. automethod:: culebra.trainer.aco.ElitistAntSystem._init_internals
.. automethod:: culebra.trainer.aco.ElitistAntSystem._reset_internals
.. automethod:: culebra.trainer.aco.ElitistAntSystem._init_search
.. automethod:: culebra.trainer.aco.ElitistAntSystem._search
.. automethod:: culebra.trainer.aco.ElitistAntSystem._finish_search
.. automethod:: culebra.trainer.aco.ElitistAntSystem._start_iteration
.. automethod:: culebra.trainer.aco.ElitistAntSystem._preprocess_iteration
.. automethod:: culebra.trainer.aco.ElitistAntSystem._do_iteration
.. automethod:: culebra.trainer.aco.ElitistAntSystem._postprocess_iteration
.. automethod:: culebra.trainer.aco.ElitistAntSystem._finish_iteration
.. automethod:: culebra.trainer.aco.ElitistAntSystem._do_iteration_stats
.. automethod:: culebra.trainer.aco.ElitistAntSystem._default_termination_func
.. automethod:: culebra.trainer.aco.ElitistAntSystem._termination_criterion
.. automethod:: culebra.trainer.aco.ElitistAntSystem._init_representatives
.. automethod:: culebra.trainer.aco.ElitistAntSystem._calculate_choice_info
.. automethod:: culebra.trainer.aco.ElitistAntSystem._initial_choice
.. automethod:: culebra.trainer.aco.ElitistAntSystem._next_choice
.. automethod:: culebra.trainer.aco.ElitistAntSystem._feasible_neighborhood_probs
.. automethod:: culebra.trainer.aco.ElitistAntSystem._generate_ant
.. automethod:: culebra.trainer.aco.ElitistAntSystem._generate_col
.. automethod:: culebra.trainer.aco.ElitistAntSystem._deposit_pheromones
.. automethod:: culebra.trainer.aco.ElitistAntSystem._reset_pheromones
.. automethod:: culebra.trainer.aco.ElitistAntSystem._increase_pheromones
.. automethod:: culebra.trainer.aco.ElitistAntSystem._decrease_pheromones
.. automethod:: culebra.trainer.aco.ElitistAntSystem._update_pheromones
.. automethod:: culebra.trainer.aco.ElitistAntSystem._update_elite
.. automethod:: culebra.trainer.aco.ElitistAntSystem._has_converged
