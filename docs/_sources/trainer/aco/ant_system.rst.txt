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

:class:`culebra.trainer.aco.AntSystem` class
============================================

.. autoclass:: culebra.trainer.aco.AntSystem

Class methods
-------------
.. automethod:: culebra.trainer.aco.AntSystem.load

Properties
----------
.. autoproperty:: culebra.trainer.aco.AntSystem.checkpoint_activation
.. autoproperty:: culebra.trainer.aco.AntSystem.checkpoint_basename
.. autoproperty:: culebra.trainer.aco.AntSystem.checkpoint_filename
.. autoproperty:: culebra.trainer.aco.AntSystem.checkpoint_freq
.. autoproperty:: culebra.trainer.aco.AntSystem.choice_info
.. autoproperty:: culebra.trainer.aco.AntSystem.col
.. autoproperty:: culebra.trainer.aco.AntSystem.col_size
.. autoproperty:: culebra.trainer.aco.AntSystem.container
.. autoproperty:: culebra.trainer.aco.AntSystem.cooperative_fitness_estimation_func
.. autoproperty:: culebra.trainer.aco.AntSystem.cooperators
.. autoproperty:: culebra.trainer.aco.AntSystem.current_iter
.. autoproperty:: culebra.trainer.aco.AntSystem.custom_termination_func
.. autoproperty:: culebra.trainer.aco.AntSystem.exploitation_prob
.. autoproperty:: culebra.trainer.aco.AntSystem.fitness_func
.. autoproperty:: culebra.trainer.aco.AntSystem.heuristic
.. autoproperty:: culebra.trainer.aco.AntSystem.heuristic_influence
.. autoproperty:: culebra.trainer.aco.AntSystem.heuristic_shapes
.. autoproperty:: culebra.trainer.aco.AntSystem.index
.. autoproperty:: culebra.trainer.aco.AntSystem.initial_pheromone
.. autoproperty:: culebra.trainer.aco.AntSystem.iteration_metric_names
.. autoproperty:: culebra.trainer.aco.AntSystem.iteration_obj_stats
.. autoproperty:: culebra.trainer.aco.AntSystem.logbook
.. autoproperty:: culebra.trainer.aco.AntSystem.max_num_iters
.. autoproperty:: culebra.trainer.aco.AntSystem.num_evals
.. autoproperty:: culebra.trainer.aco.AntSystem.num_iters
.. autoproperty:: culebra.trainer.aco.AntSystem.num_heuristic_matrices
.. autoproperty:: culebra.trainer.aco.AntSystem.num_pheromone_matrices
.. autoproperty:: culebra.trainer.aco.AntSystem.pheromone
.. autoproperty:: culebra.trainer.aco.AntSystem.pheromone_evaporation_rate
.. autoproperty:: culebra.trainer.aco.AntSystem.pheromone_influence
.. autoproperty:: culebra.trainer.aco.AntSystem.pheromone_shapes
.. autoproperty:: culebra.trainer.aco.AntSystem.random_seed
.. autoproperty:: culebra.trainer.aco.AntSystem.receive_representatives_func
.. autoproperty:: culebra.trainer.aco.AntSystem.runtime
.. autoproperty:: culebra.trainer.aco.AntSystem.send_representatives_func
.. autoproperty:: culebra.trainer.aco.AntSystem.solution_cls
.. autoproperty:: culebra.trainer.aco.AntSystem.species
.. autoproperty:: culebra.trainer.aco.AntSystem.state_proxy
.. autoproperty:: culebra.trainer.aco.AntSystem.training_finished
.. autoproperty:: culebra.trainer.aco.AntSystem.verbosity

Private properties
------------------
.. autoproperty:: culebra.trainer.aco.AntSystem._default_checkpoint_activation
.. autoproperty:: culebra.trainer.aco.AntSystem._default_checkpoint_basename
.. autoproperty:: culebra.trainer.aco.AntSystem._default_checkpoint_freq
.. autoproperty:: culebra.trainer.aco.AntSystem._default_col_size
.. autoproperty:: culebra.trainer.aco.AntSystem._default_cooperative_fitness_estimation_func
.. autoproperty:: culebra.trainer.aco.AntSystem._default_exploitation_prob
.. autoproperty:: culebra.trainer.aco.AntSystem._default_heuristic
.. autoproperty:: culebra.trainer.aco.AntSystem._default_heuristic_influence
.. autoproperty:: culebra.trainer.aco.AntSystem._default_index
.. autoproperty:: culebra.trainer.aco.AntSystem._default_max_num_iters
.. autoproperty:: culebra.trainer.aco.AntSystem._default_pheromone_evaporation_rate
.. autoproperty:: culebra.trainer.aco.AntSystem._default_pheromone_influence
.. autoproperty:: culebra.trainer.aco.AntSystem._default_receive_representatives_func
.. autoproperty:: culebra.trainer.aco.AntSystem._default_send_representatives_func
.. autoproperty:: culebra.trainer.aco.AntSystem._default_verbosity

Methods
-------
.. automethod:: culebra.trainer.aco.AntSystem.best_cooperators
.. automethod:: culebra.trainer.aco.AntSystem.best_solutions
.. automethod:: culebra.trainer.aco.AntSystem.dump
.. automethod:: culebra.trainer.aco.AntSystem.evaluate
.. automethod:: culebra.trainer.aco.AntSystem.integrate_representatives
.. automethod:: culebra.trainer.aco.AntSystem.reset
.. automethod:: culebra.trainer.aco.AntSystem.select_representatives
.. automethod:: culebra.trainer.aco.AntSystem.test
.. automethod:: culebra.trainer.aco.AntSystem.train

Private methods
---------------
.. automethod:: culebra.trainer.aco.AntSystem._ant_choice_info
.. automethod:: culebra.trainer.aco.AntSystem._calculate_choice_info
.. automethod:: culebra.trainer.aco.AntSystem._decrease_pheromone
.. automethod:: culebra.trainer.aco.AntSystem._default_termination_func
.. automethod:: culebra.trainer.aco.AntSystem._deposit_pheromone
.. automethod:: culebra.trainer.aco.AntSystem._do_iteration
.. automethod:: culebra.trainer.aco.AntSystem._do_training
.. automethod:: culebra.trainer.aco.AntSystem._finish_iteration
.. automethod:: culebra.trainer.aco.AntSystem._finish_training
.. automethod:: culebra.trainer.aco.AntSystem._generate_ant
.. automethod:: culebra.trainer.aco.AntSystem._generate_col
.. automethod:: culebra.trainer.aco.AntSystem._generate_cooperators
.. automethod:: culebra.trainer.aco.AntSystem._get_iteration_metrics
.. automethod:: culebra.trainer.aco.AntSystem._get_objective_stats
.. automethod:: culebra.trainer.aco.AntSystem._get_state
.. automethod:: culebra.trainer.aco.AntSystem._increase_pheromone
.. automethod:: culebra.trainer.aco.AntSystem._init_internals
.. automethod:: culebra.trainer.aco.AntSystem._init_pheromone
.. automethod:: culebra.trainer.aco.AntSystem._init_state
.. automethod:: culebra.trainer.aco.AntSystem._init_training
.. automethod:: culebra.trainer.aco.AntSystem._load_state
.. automethod:: culebra.trainer.aco.AntSystem._new_state
.. automethod:: culebra.trainer.aco.AntSystem._next_choice
.. automethod:: culebra.trainer.aco.AntSystem._pheromone_amount
.. automethod:: culebra.trainer.aco.AntSystem._reset_internals
.. automethod:: culebra.trainer.aco.AntSystem._reset_state
.. automethod:: culebra.trainer.aco.AntSystem._save_state
.. automethod:: culebra.trainer.aco.AntSystem._set_state
.. automethod:: culebra.trainer.aco.AntSystem._start_iteration
.. automethod:: culebra.trainer.aco.AntSystem._termination_criterion
.. automethod:: culebra.trainer.aco.AntSystem._unfeasible_nodes
.. automethod:: culebra.trainer.aco.AntSystem._update_logbook
.. automethod:: culebra.trainer.aco.AntSystem._update_pheromone
