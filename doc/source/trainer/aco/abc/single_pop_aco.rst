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

:py:class:`culebra.trainer.aco.abc.SinglePopACO` class
======================================================

.. autoclass:: culebra.trainer.aco.abc.SinglePopACO

Class attributes
----------------
.. autoattribute:: culebra.trainer.aco.abc.SinglePopACO.stats_names
.. autoattribute:: culebra.trainer.aco.abc.SinglePopACO.objective_stats

Properties
----------
.. autoproperty:: culebra.trainer.aco.abc.SinglePopACO.solution_cls
.. autoproperty:: culebra.trainer.aco.abc.SinglePopACO.species
.. autoproperty:: culebra.trainer.aco.abc.SinglePopACO.fitness_function
.. autoproperty:: culebra.trainer.aco.abc.SinglePopACO.initial_pheromones
.. autoproperty:: culebra.trainer.aco.abc.SinglePopACO.heuristics
.. autoproperty:: culebra.trainer.aco.abc.SinglePopACO.pheromones
.. autoproperty:: culebra.trainer.aco.abc.SinglePopACO.choice_info
.. autoproperty:: culebra.trainer.aco.abc.SinglePopACO.max_num_iters
.. autoproperty:: culebra.trainer.aco.abc.SinglePopACO.current_iter
.. autoproperty:: culebra.trainer.aco.abc.SinglePopACO.custom_termination_func
.. autoproperty:: culebra.trainer.aco.abc.SinglePopACO.pop_size
.. autoproperty:: culebra.trainer.aco.abc.SinglePopACO.checkpoint_enable
.. autoproperty:: culebra.trainer.aco.abc.SinglePopACO.checkpoint_freq
.. autoproperty:: culebra.trainer.aco.abc.SinglePopACO.checkpoint_filename
.. autoproperty:: culebra.trainer.aco.abc.SinglePopACO.verbose
.. autoproperty:: culebra.trainer.aco.abc.SinglePopACO.random_seed
.. autoproperty:: culebra.trainer.aco.abc.SinglePopACO.logbook
.. autoproperty:: culebra.trainer.aco.abc.SinglePopACO.num_evals
.. autoproperty:: culebra.trainer.aco.abc.SinglePopACO.runtime
.. autoproperty:: culebra.trainer.aco.abc.SinglePopACO.index
.. autoproperty:: culebra.trainer.aco.abc.SinglePopACO.container
.. autoproperty:: culebra.trainer.aco.abc.SinglePopACO.representatives
.. autoproperty:: culebra.trainer.aco.abc.SinglePopACO.pop

Private properties
------------------
.. autoproperty:: culebra.trainer.aco.abc.SinglePopACO._state

Methods
-------
.. automethod:: culebra.trainer.aco.abc.SinglePopACO.reset
.. automethod:: culebra.trainer.aco.abc.SinglePopACO.evaluate
.. automethod:: culebra.trainer.aco.abc.SinglePopACO.best_solutions
.. automethod:: culebra.trainer.aco.abc.SinglePopACO.best_representatives
.. automethod:: culebra.trainer.aco.abc.SinglePopACO.train
.. automethod:: culebra.trainer.aco.abc.SinglePopACO.test

Private methods
---------------
.. automethod:: culebra.trainer.aco.abc.SinglePopACO._save_state
.. automethod:: culebra.trainer.aco.abc.SinglePopACO._load_state
.. automethod:: culebra.trainer.aco.abc.SinglePopACO._new_state
.. automethod:: culebra.trainer.aco.abc.SinglePopACO._init_state
.. automethod:: culebra.trainer.aco.abc.SinglePopACO._reset_state
.. automethod:: culebra.trainer.aco.abc.SinglePopACO._init_internals
.. automethod:: culebra.trainer.aco.abc.SinglePopACO._reset_internals
.. automethod:: culebra.trainer.aco.abc.SinglePopACO._init_search
.. automethod:: culebra.trainer.aco.abc.SinglePopACO._search
.. automethod:: culebra.trainer.aco.abc.SinglePopACO._finish_search
.. automethod:: culebra.trainer.aco.abc.SinglePopACO._start_iteration
.. automethod:: culebra.trainer.aco.abc.SinglePopACO._preprocess_iteration
.. automethod:: culebra.trainer.aco.abc.SinglePopACO._do_iteration
.. automethod:: culebra.trainer.aco.abc.SinglePopACO._postprocess_iteration
.. automethod:: culebra.trainer.aco.abc.SinglePopACO._finish_iteration
.. automethod:: culebra.trainer.aco.abc.SinglePopACO._do_iteration_stats
.. automethod:: culebra.trainer.aco.abc.SinglePopACO._default_termination_func
.. automethod:: culebra.trainer.aco.abc.SinglePopACO._termination_criterion
.. automethod:: culebra.trainer.aco.abc.SinglePopACO._init_representatives
.. automethod:: culebra.trainer.aco.abc.SinglePopACO._calculate_choice_info
.. automethod:: culebra.trainer.aco.abc.SinglePopACO._initial_choice
.. automethod:: culebra.trainer.aco.abc.SinglePopACO._next_choice
.. automethod:: culebra.trainer.aco.abc.SinglePopACO._feasible_neighborhood_probs
.. automethod:: culebra.trainer.aco.abc.SinglePopACO._generate_ant
.. automethod:: culebra.trainer.aco.abc.SinglePopACO._generate_pop
.. automethod:: culebra.trainer.aco.abc.SinglePopACO._evaporate_pheromones
.. automethod:: culebra.trainer.aco.abc.SinglePopACO._deposit_pheromones
.. automethod:: culebra.trainer.aco.abc.SinglePopACO._update_pheromones

