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

:class:`culebra.trainer.aco.abc.ACO` class
==========================================

.. autoclass:: culebra.trainer.aco.abc.ACO

Class methods
-------------
.. automethod:: culebra.trainer.aco.abc.ACO.load

Properties
----------
.. autoproperty:: culebra.trainer.aco.abc.ACO.checkpoint_activation
.. autoproperty:: culebra.trainer.aco.abc.ACO.checkpoint_basename
.. autoproperty:: culebra.trainer.aco.abc.ACO.checkpoint_filename
.. autoproperty:: culebra.trainer.aco.abc.ACO.checkpoint_freq
.. autoproperty:: culebra.trainer.aco.abc.ACO.choice_info
.. autoproperty:: culebra.trainer.aco.abc.ACO.col
.. autoproperty:: culebra.trainer.aco.abc.ACO.col_size
.. autoproperty:: culebra.trainer.aco.abc.ACO.container
.. autoproperty:: culebra.trainer.aco.abc.ACO.cooperative_fitness_estimation_func
.. autoproperty:: culebra.trainer.aco.abc.ACO.cooperators
.. autoproperty:: culebra.trainer.aco.abc.ACO.current_iter
.. autoproperty:: culebra.trainer.aco.abc.ACO.custom_termination_func
.. autoproperty:: culebra.trainer.aco.abc.ACO.exploitation_prob
.. autoproperty:: culebra.trainer.aco.abc.ACO.fitness_func
.. autoproperty:: culebra.trainer.aco.abc.ACO.heuristic
.. autoproperty:: culebra.trainer.aco.abc.ACO.heuristic_influence
.. autoproperty:: culebra.trainer.aco.abc.ACO.heuristic_shapes
.. autoproperty:: culebra.trainer.aco.abc.ACO.index
.. autoproperty:: culebra.trainer.aco.abc.ACO.initial_pheromone
.. autoproperty:: culebra.trainer.aco.abc.ACO.iteration_metric_names
.. autoproperty:: culebra.trainer.aco.abc.ACO.iteration_obj_stats
.. autoproperty:: culebra.trainer.aco.abc.ACO.logbook
.. autoproperty:: culebra.trainer.aco.abc.ACO.max_num_iters
.. autoproperty:: culebra.trainer.aco.abc.ACO.num_evals
.. autoproperty:: culebra.trainer.aco.abc.ACO.num_iters
.. autoproperty:: culebra.trainer.aco.abc.ACO.num_heuristic_matrices
.. autoproperty:: culebra.trainer.aco.abc.ACO.num_pheromone_matrices
.. autoproperty:: culebra.trainer.aco.abc.ACO.pheromone
.. autoproperty:: culebra.trainer.aco.abc.ACO.pheromone_influence
.. autoproperty:: culebra.trainer.aco.abc.ACO.pheromone_shapes
.. autoproperty:: culebra.trainer.aco.abc.ACO.random_seed
.. autoproperty:: culebra.trainer.aco.abc.ACO.receive_representatives_func
.. autoproperty:: culebra.trainer.aco.abc.ACO.runtime
.. autoproperty:: culebra.trainer.aco.abc.ACO.send_representatives_func
.. autoproperty:: culebra.trainer.aco.abc.ACO.solution_cls
.. autoproperty:: culebra.trainer.aco.abc.ACO.species
.. autoproperty:: culebra.trainer.aco.abc.ACO.state_proxy
.. autoproperty:: culebra.trainer.aco.abc.ACO.training_finished
.. autoproperty:: culebra.trainer.aco.abc.ACO.verbosity

Private properties
------------------
.. autoproperty:: culebra.trainer.aco.abc.ACO._default_checkpoint_activation
.. autoproperty:: culebra.trainer.aco.abc.ACO._default_checkpoint_basename
.. autoproperty:: culebra.trainer.aco.abc.ACO._default_checkpoint_freq
.. autoproperty:: culebra.trainer.aco.abc.ACO._default_col_size
.. autoproperty:: culebra.trainer.aco.abc.ACO._default_cooperative_fitness_estimation_func
.. autoproperty:: culebra.trainer.aco.abc.ACO._default_exploitation_prob
.. autoproperty:: culebra.trainer.aco.abc.ACO._default_heuristic
.. autoproperty:: culebra.trainer.aco.abc.ACO._default_heuristic_influence
.. autoproperty:: culebra.trainer.aco.abc.ACO._default_index
.. autoproperty:: culebra.trainer.aco.abc.ACO._default_max_num_iters
.. autoproperty:: culebra.trainer.aco.abc.ACO._default_pheromone_influence
.. autoproperty:: culebra.trainer.aco.abc.ACO._default_receive_representatives_func
.. autoproperty:: culebra.trainer.aco.abc.ACO._default_send_representatives_func
.. autoproperty:: culebra.trainer.aco.abc.ACO._default_verbosity

Methods
-------
.. automethod:: culebra.trainer.aco.abc.ACO.best_cooperators
.. automethod:: culebra.trainer.aco.abc.ACO.best_solutions
.. automethod:: culebra.trainer.aco.abc.ACO.dump
.. automethod:: culebra.trainer.aco.abc.ACO.evaluate
.. automethod:: culebra.trainer.aco.abc.ACO.integrate_representatives
.. automethod:: culebra.trainer.aco.abc.ACO.reset
.. automethod:: culebra.trainer.aco.abc.ACO.select_representatives
.. automethod:: culebra.trainer.aco.abc.ACO.test
.. automethod:: culebra.trainer.aco.abc.ACO.train

Private methods
---------------
.. automethod:: culebra.trainer.aco.abc.ACO._ant_choice_info
.. automethod:: culebra.trainer.aco.abc.ACO._calculate_choice_info
.. automethod:: culebra.trainer.aco.abc.ACO._default_termination_func
.. automethod:: culebra.trainer.aco.abc.ACO._deposit_pheromone
.. automethod:: culebra.trainer.aco.abc.ACO._do_iteration
.. automethod:: culebra.trainer.aco.abc.ACO._do_training
.. automethod:: culebra.trainer.aco.abc.ACO._finish_iteration
.. automethod:: culebra.trainer.aco.abc.ACO._finish_training
.. automethod:: culebra.trainer.aco.abc.ACO._generate_ant
.. automethod:: culebra.trainer.aco.abc.ACO._generate_col
.. automethod:: culebra.trainer.aco.abc.ACO._generate_cooperators
.. automethod:: culebra.trainer.aco.abc.ACO._get_iteration_metrics
.. automethod:: culebra.trainer.aco.abc.ACO._get_objective_stats
.. automethod:: culebra.trainer.aco.abc.ACO._get_state
.. automethod:: culebra.trainer.aco.abc.ACO._init_internals
.. automethod:: culebra.trainer.aco.abc.ACO._init_pheromone
.. automethod:: culebra.trainer.aco.abc.ACO._init_state
.. automethod:: culebra.trainer.aco.abc.ACO._init_training
.. automethod:: culebra.trainer.aco.abc.ACO._load_state
.. automethod:: culebra.trainer.aco.abc.ACO._new_state
.. automethod:: culebra.trainer.aco.abc.ACO._next_choice
.. automethod:: culebra.trainer.aco.abc.ACO._pheromone_amount
.. automethod:: culebra.trainer.aco.abc.ACO._reset_internals
.. automethod:: culebra.trainer.aco.abc.ACO._reset_state
.. automethod:: culebra.trainer.aco.abc.ACO._save_state
.. automethod:: culebra.trainer.aco.abc.ACO._set_state
.. automethod:: culebra.trainer.aco.abc.ACO._start_iteration
.. automethod:: culebra.trainer.aco.abc.ACO._termination_criterion
.. automethod:: culebra.trainer.aco.abc.ACO._update_logbook
.. automethod:: culebra.trainer.aco.abc.ACO._update_pheromone