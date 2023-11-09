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

:py:class:`culebra.trainer.aco.abc.SingleObjACO` class
======================================================

.. autoclass:: culebra.trainer.aco.abc.SingleObjACO

Class attributes
----------------
.. autoattribute:: culebra.trainer.aco.abc.SingleObjACO.stats_names
.. autoattribute:: culebra.trainer.aco.abc.SingleObjACO.objective_stats

Properties
----------
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.solution_cls
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.species
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.fitness_function
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.initial_pheromones
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.heuristics
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.pheromones_influence
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.heuristics_influence
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.pheromones
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.choice_info
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.max_num_iters
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.current_iter
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.custom_termination_func
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.col_size
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.checkpoint_enable
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.checkpoint_freq
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.checkpoint_filename
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.verbose
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.random_seed
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.logbook
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.num_evals
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.runtime
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.index
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.container
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.representatives
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO.col

Private properties
------------------
.. autoproperty:: culebra.trainer.aco.abc.SingleObjACO._state

Methods
-------
.. automethod:: culebra.trainer.aco.abc.SingleObjACO.reset
.. automethod:: culebra.trainer.aco.abc.SingleObjACO.evaluate
.. automethod:: culebra.trainer.aco.abc.SingleObjACO.best_solutions
.. automethod:: culebra.trainer.aco.abc.SingleObjACO.best_representatives
.. automethod:: culebra.trainer.aco.abc.SingleObjACO.train
.. automethod:: culebra.trainer.aco.abc.SingleObjACO.test

Private methods
---------------
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._save_state
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._load_state
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._new_state
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._init_state
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._reset_state
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._init_internals
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._reset_internals
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._init_search
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._search
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._finish_search
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._start_iteration
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._preprocess_iteration
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._do_iteration
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._postprocess_iteration
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._finish_iteration
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._do_iteration_stats
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._default_termination_func
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._termination_criterion
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._init_representatives
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._calculate_choice_info
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._initial_choice
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._next_choice
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._feasible_neighborhood_probs
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._generate_ant
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._generate_col
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._deposit_pheromones
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._increase_pheromones
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._decrease_pheromones
.. automethod:: culebra.trainer.aco.abc.SingleObjACO._update_pheromones
