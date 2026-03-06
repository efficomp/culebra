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

:class:`culebra.trainer.aco.MMAS` class
=======================================

.. autoclass:: culebra.trainer.aco.MMAS

Class methods
-------------
.. automethod:: culebra.trainer.aco.MMAS.load

Properties
----------
.. autoproperty:: culebra.trainer.aco.MMAS.checkpoint_activation
.. autoproperty:: culebra.trainer.aco.MMAS.checkpoint_basename
.. autoproperty:: culebra.trainer.aco.MMAS.checkpoint_filename
.. autoproperty:: culebra.trainer.aco.MMAS.checkpoint_freq
.. autoproperty:: culebra.trainer.aco.MMAS.choice_info
.. autoproperty:: culebra.trainer.aco.MMAS.col
.. autoproperty:: culebra.trainer.aco.MMAS.col_size
.. autoproperty:: culebra.trainer.aco.MMAS.container
.. autoproperty:: culebra.trainer.aco.MMAS.convergence_check_freq
.. autoproperty:: culebra.trainer.aco.MMAS.cooperative_fitness_estimation_func
.. autoproperty:: culebra.trainer.aco.MMAS.cooperators
.. autoproperty:: culebra.trainer.aco.MMAS.current_iter
.. autoproperty:: culebra.trainer.aco.MMAS.custom_termination_func
.. autoproperty:: culebra.trainer.aco.MMAS.exploitation_prob
.. autoproperty:: culebra.trainer.aco.MMAS.fitness_func
.. autoproperty:: culebra.trainer.aco.MMAS.heuristic
.. autoproperty:: culebra.trainer.aco.MMAS.heuristic_influence
.. autoproperty:: culebra.trainer.aco.MMAS.heuristic_shapes
.. autoproperty:: culebra.trainer.aco.MMAS.index
.. autoproperty:: culebra.trainer.aco.MMAS.initial_pheromone
.. autoproperty:: culebra.trainer.aco.MMAS.iter_best_use_limit
.. autoproperty:: culebra.trainer.aco.MMAS.iteration_metric_names
.. autoproperty:: culebra.trainer.aco.MMAS.iteration_obj_stats
.. autoproperty:: culebra.trainer.aco.MMAS.logbook
.. autoproperty:: culebra.trainer.aco.MMAS.max_num_iters
.. autoproperty:: culebra.trainer.aco.MMAS.num_evals
.. autoproperty:: culebra.trainer.aco.MMAS.num_iters
.. autoproperty:: culebra.trainer.aco.MMAS.num_heuristic_matrices
.. autoproperty:: culebra.trainer.aco.MMAS.num_pheromone_matrices
.. autoproperty:: culebra.trainer.aco.MMAS.pheromone
.. autoproperty:: culebra.trainer.aco.MMAS.pheromone_evaporation_rate
.. autoproperty:: culebra.trainer.aco.MMAS.pheromone_influence
.. autoproperty:: culebra.trainer.aco.MMAS.pheromone_shapes
.. autoproperty:: culebra.trainer.aco.MMAS.random_seed
.. autoproperty:: culebra.trainer.aco.MMAS.receive_representatives_func
.. autoproperty:: culebra.trainer.aco.MMAS.runtime
.. autoproperty:: culebra.trainer.aco.MMAS.send_representatives_func
.. autoproperty:: culebra.trainer.aco.MMAS.solution_cls
.. autoproperty:: culebra.trainer.aco.MMAS.species
.. autoproperty:: culebra.trainer.aco.MMAS.state_proxy
.. autoproperty:: culebra.trainer.aco.MMAS.training_finished
.. autoproperty:: culebra.trainer.aco.MMAS.verbosity

Private properties
------------------
.. autoproperty:: culebra.trainer.aco.MMAS._default_checkpoint_activation
.. autoproperty:: culebra.trainer.aco.MMAS._default_checkpoint_basename
.. autoproperty:: culebra.trainer.aco.MMAS._default_checkpoint_freq
.. autoproperty:: culebra.trainer.aco.MMAS._default_col_size
.. autoproperty:: culebra.trainer.aco.MMAS._default_convergence_check_freq
.. autoproperty:: culebra.trainer.aco.MMAS._default_cooperative_fitness_estimation_func
.. autoproperty:: culebra.trainer.aco.MMAS._default_exploitation_prob
.. autoproperty:: culebra.trainer.aco.MMAS._default_heuristic
.. autoproperty:: culebra.trainer.aco.MMAS._default_heuristic_influence
.. autoproperty:: culebra.trainer.aco.MMAS._default_index
.. autoproperty:: culebra.trainer.aco.MMAS._default_iter_best_use_limit
.. autoproperty:: culebra.trainer.aco.MMAS._default_max_num_iters
.. autoproperty:: culebra.trainer.aco.MMAS._default_pheromone_evaporation_rate
.. autoproperty:: culebra.trainer.aco.MMAS._default_pheromone_influence
.. autoproperty:: culebra.trainer.aco.MMAS._default_receive_representatives_func
.. autoproperty:: culebra.trainer.aco.MMAS._default_send_representatives_func
.. autoproperty:: culebra.trainer.aco.MMAS._default_verbosity
.. autoproperty:: culebra.trainer.aco.MMAS._global_best_freq

Methods
-------
.. automethod:: culebra.trainer.aco.MMAS.best_cooperators
.. automethod:: culebra.trainer.aco.MMAS.best_solutions
.. automethod:: culebra.trainer.aco.MMAS.dump
.. automethod:: culebra.trainer.aco.MMAS.evaluate
.. automethod:: culebra.trainer.aco.MMAS.integrate_representatives
.. automethod:: culebra.trainer.aco.MMAS.reset
.. automethod:: culebra.trainer.aco.MMAS.select_representatives
.. automethod:: culebra.trainer.aco.MMAS.test
.. automethod:: culebra.trainer.aco.MMAS.train

Private methods
---------------
.. automethod:: culebra.trainer.aco.MMAS._ant_choice_info
.. automethod:: culebra.trainer.aco.MMAS._calculate_choice_info
.. automethod:: culebra.trainer.aco.MMAS._decrease_pheromone
.. automethod:: culebra.trainer.aco.MMAS._default_termination_func
.. automethod:: culebra.trainer.aco.MMAS._deposit_pheromone
.. automethod:: culebra.trainer.aco.MMAS._do_iteration
.. automethod:: culebra.trainer.aco.MMAS._do_training
.. automethod:: culebra.trainer.aco.MMAS._finish_iteration
.. automethod:: culebra.trainer.aco.MMAS._finish_training
.. automethod:: culebra.trainer.aco.MMAS._generate_ant
.. automethod:: culebra.trainer.aco.MMAS._generate_col
.. automethod:: culebra.trainer.aco.MMAS._generate_cooperators
.. automethod:: culebra.trainer.aco.MMAS._get_iteration_metrics
.. automethod:: culebra.trainer.aco.MMAS._get_objective_stats
.. automethod:: culebra.trainer.aco.MMAS._get_state
.. automethod:: culebra.trainer.aco.MMAS._has_converged
.. automethod:: culebra.trainer.aco.MMAS._increase_pheromone
.. automethod:: culebra.trainer.aco.MMAS._init_internals
.. automethod:: culebra.trainer.aco.MMAS._init_pheromone
.. automethod:: culebra.trainer.aco.MMAS._init_state
.. automethod:: culebra.trainer.aco.MMAS._init_training
.. automethod:: culebra.trainer.aco.MMAS._load_state
.. automethod:: culebra.trainer.aco.MMAS._new_state
.. automethod:: culebra.trainer.aco.MMAS._next_choice
.. automethod:: culebra.trainer.aco.MMAS._pheromone_amount
.. automethod:: culebra.trainer.aco.MMAS._reset_internals
.. automethod:: culebra.trainer.aco.MMAS._reset_state
.. automethod:: culebra.trainer.aco.MMAS._save_state
.. automethod:: culebra.trainer.aco.MMAS._set_state
.. automethod:: culebra.trainer.aco.MMAS._should_reset_pheromone
.. automethod:: culebra.trainer.aco.MMAS._start_iteration
.. automethod:: culebra.trainer.aco.MMAS._termination_criterion
.. automethod:: culebra.trainer.aco.MMAS._update_elite
.. automethod:: culebra.trainer.aco.MMAS._update_logbook
.. automethod:: culebra.trainer.aco.MMAS._update_pheromone