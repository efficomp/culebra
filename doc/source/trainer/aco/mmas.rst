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

:py:class:`culebra.trainer.aco.MMAS` class
==========================================

.. autoclass:: culebra.trainer.aco.MMAS

Class attributes
----------------

.. autoattribute:: culebra.trainer.aco.MMAS.stats_names
.. autoattribute:: culebra.trainer.aco.MMAS.objective_stats

Properties
----------

.. autoproperty:: culebra.trainer.aco.MMAS.solution_cls
.. autoproperty:: culebra.trainer.aco.MMAS.species
.. autoproperty:: culebra.trainer.aco.MMAS.fitness_function
.. autoproperty:: culebra.trainer.aco.MMAS.initial_pheromones
.. autoproperty:: culebra.trainer.aco.MMAS.heuristics
.. autoproperty:: culebra.trainer.aco.MMAS.pheromone_influence
.. autoproperty:: culebra.trainer.aco.MMAS.heuristic_influence
.. autoproperty:: culebra.trainer.aco.MMAS.pheromone_evaporation_rate
.. autoproperty:: culebra.trainer.aco.MMAS.elite_weight
.. autoproperty:: culebra.trainer.aco.MMAS.iter_best_use_limit
.. autoproperty:: culebra.trainer.aco.MMAS.convergence_check_freq
.. autoproperty:: culebra.trainer.aco.MMAS.pheromones
.. autoproperty:: culebra.trainer.aco.MMAS.choice_info
.. autoproperty:: culebra.trainer.aco.MMAS.max_num_iters
.. autoproperty:: culebra.trainer.aco.MMAS.current_iter
.. autoproperty:: culebra.trainer.aco.MMAS.custom_termination_func
.. autoproperty:: culebra.trainer.aco.MMAS.pop_size
.. autoproperty:: culebra.trainer.aco.MMAS.checkpoint_enable
.. autoproperty:: culebra.trainer.aco.MMAS.checkpoint_freq
.. autoproperty:: culebra.trainer.aco.MMAS.checkpoint_filename
.. autoproperty:: culebra.trainer.aco.MMAS.verbose
.. autoproperty:: culebra.trainer.aco.MMAS.random_seed
.. autoproperty:: culebra.trainer.aco.MMAS.logbook
.. autoproperty:: culebra.trainer.aco.MMAS.num_evals
.. autoproperty:: culebra.trainer.aco.MMAS.runtime
.. autoproperty:: culebra.trainer.aco.MMAS.index
.. autoproperty:: culebra.trainer.aco.MMAS.container
.. autoproperty:: culebra.trainer.aco.MMAS.representatives
.. autoproperty:: culebra.trainer.aco.MMAS.pop


Private properties
------------------

.. autoproperty:: culebra.trainer.aco.MMAS._state
.. autoproperty:: culebra.trainer.aco.MMAS._global_best_freq

Methods
-------

.. automethod:: culebra.trainer.aco.MMAS.reset
.. automethod:: culebra.trainer.aco.MMAS.evaluate
.. automethod:: culebra.trainer.aco.MMAS.best_solutions
.. automethod:: culebra.trainer.aco.MMAS.best_representatives
.. automethod:: culebra.trainer.aco.MMAS.train
.. automethod:: culebra.trainer.aco.MMAS.test

Private methods
---------------

.. automethod:: culebra.trainer.aco.MMAS._save_state
.. automethod:: culebra.trainer.aco.MMAS._load_state
.. automethod:: culebra.trainer.aco.MMAS._new_state
.. automethod:: culebra.trainer.aco.MMAS._init_state
.. automethod:: culebra.trainer.aco.MMAS._reset_state
.. automethod:: culebra.trainer.aco.MMAS._init_internals
.. automethod:: culebra.trainer.aco.MMAS._reset_internals
.. automethod:: culebra.trainer.aco.MMAS._init_search
.. automethod:: culebra.trainer.aco.MMAS._search
.. automethod:: culebra.trainer.aco.MMAS._finish_search
.. automethod:: culebra.trainer.aco.MMAS._start_iteration
.. automethod:: culebra.trainer.aco.MMAS._preprocess_iteration
.. automethod:: culebra.trainer.aco.MMAS._do_iteration
.. automethod:: culebra.trainer.aco.MMAS._postprocess_iteration
.. automethod:: culebra.trainer.aco.MMAS._finish_iteration
.. automethod:: culebra.trainer.aco.MMAS._do_iteration_stats
.. automethod:: culebra.trainer.aco.MMAS._default_termination_func
.. automethod:: culebra.trainer.aco.MMAS._termination_criterion
.. automethod:: culebra.trainer.aco.MMAS._init_representatives
.. automethod:: culebra.trainer.aco.MMAS._calculate_choice_info
.. automethod:: culebra.trainer.aco.MMAS._initial_choice
.. automethod:: culebra.trainer.aco.MMAS._next_choice
.. automethod:: culebra.trainer.aco.MMAS._feasible_neighborhood_probs
.. automethod:: culebra.trainer.aco.MMAS._generate_ant
.. automethod:: culebra.trainer.aco.MMAS._generate_pop
.. automethod:: culebra.trainer.aco.MMAS._evaporate_pheromones
.. automethod:: culebra.trainer.aco.MMAS._deposit_pop_pheromones
.. automethod:: culebra.trainer.aco.MMAS._deposit_pheromones
.. automethod:: culebra.trainer.aco.MMAS._update_pheromones
.. automethod:: culebra.trainer.aco.MMAS._update_elite
.. automethod:: culebra.trainer.aco.MMAS._has_converged
