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

:class:`culebra.trainer.aco.abc.ACOFS2D` class
==============================================

.. autoclass:: culebra.trainer.aco.abc.ACOFS2D

Class methods
-------------
.. automethod:: culebra.trainer.aco.abc.ACOFS2D.load

Properties
----------
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D.checkpoint_activation
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D.checkpoint_basename
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D.checkpoint_filename
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D.checkpoint_freq
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D.choice_info
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D.col
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D.col_size
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D.container
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D.cooperative_fitness_estimation_func
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D.cooperators
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D.current_iter
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D.custom_termination_func
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D.discard_prob
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D.exploitation_prob
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D.fitness_func
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D.heuristic
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D.heuristic_influence
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D.heuristic_shapes
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D.index
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D.initial_pheromone
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D.iteration_metric_names
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D.iteration_obj_stats
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D.logbook
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D.max_num_iters
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D.num_evals
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D.num_iters
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D.num_heuristic_matrices
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D.num_pheromone_matrices
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D.pheromone
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D.pheromone_influence
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D.pheromone_shapes
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D.random_seed
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D.receive_representatives_func
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D.runtime
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D.send_representatives_func
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D.solution_cls
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D.species
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D.state_proxy
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D.training_finished
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D.verbosity

Private properties
------------------
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D._default_checkpoint_activation
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D._default_checkpoint_basename
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D._default_checkpoint_freq
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D._default_col_size
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D._default_cooperative_fitness_estimation_func
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D._default_discard_prob
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D._default_exploitation_prob
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D._default_heuristic
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D._default_heuristic_influence
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D._default_index
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D._default_initial_pheromone
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D._default_max_num_iters
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D._default_pheromone_influence
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D._default_receive_representatives_func
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D._default_send_representatives_func
.. autoproperty:: culebra.trainer.aco.abc.ACOFS2D._default_verbosity

Methods
-------
.. automethod:: culebra.trainer.aco.abc.ACOFS2D.best_cooperators
.. automethod:: culebra.trainer.aco.abc.ACOFS2D.best_solutions
.. automethod:: culebra.trainer.aco.abc.ACOFS2D.dump
.. automethod:: culebra.trainer.aco.abc.ACOFS2D.evaluate
.. automethod:: culebra.trainer.aco.abc.ACOFS2D.integrate_representatives
.. automethod:: culebra.trainer.aco.abc.ACOFS2D.reset
.. automethod:: culebra.trainer.aco.abc.ACOFS2D.select_representatives
.. automethod:: culebra.trainer.aco.abc.ACOFS2D.test
.. automethod:: culebra.trainer.aco.abc.ACOFS2D.train

Private methods
---------------
.. automethod:: culebra.trainer.aco.abc.ACOFS2D._ant_choice_info
.. automethod:: culebra.trainer.aco.abc.ACOFS2D._calculate_choice_info
.. automethod:: culebra.trainer.aco.abc.ACOFS2D._default_termination_func
.. automethod:: culebra.trainer.aco.abc.ACOFS2D._deposit_pheromone
.. automethod:: culebra.trainer.aco.abc.ACOFS2D._do_iteration
.. automethod:: culebra.trainer.aco.abc.ACOFS2D._do_training
.. automethod:: culebra.trainer.aco.abc.ACOFS2D._finish_iteration
.. automethod:: culebra.trainer.aco.abc.ACOFS2D._finish_training
.. automethod:: culebra.trainer.aco.abc.ACOFS2D._generate_ant
.. automethod:: culebra.trainer.aco.abc.ACOFS2D._generate_col
.. automethod:: culebra.trainer.aco.abc.ACOFS2D._generate_cooperators
.. automethod:: culebra.trainer.aco.abc.ACOFS2D._get_iteration_metrics
.. automethod:: culebra.trainer.aco.abc.ACOFS2D._get_objective_stats
.. automethod:: culebra.trainer.aco.abc.ACOFS2D._get_state
.. automethod:: culebra.trainer.aco.abc.ACOFS2D._init_internals
.. automethod:: culebra.trainer.aco.abc.ACOFS2D._init_pheromone
.. automethod:: culebra.trainer.aco.abc.ACOFS2D._init_state
.. automethod:: culebra.trainer.aco.abc.ACOFS2D._init_training
.. automethod:: culebra.trainer.aco.abc.ACOFS2D._load_state
.. automethod:: culebra.trainer.aco.abc.ACOFS2D._new_state
.. automethod:: culebra.trainer.aco.abc.ACOFS2D._next_choice
.. automethod:: culebra.trainer.aco.abc.ACOFS2D._pheromone_amount
.. automethod:: culebra.trainer.aco.abc.ACOFS2D._reset_internals
.. automethod:: culebra.trainer.aco.abc.ACOFS2D._reset_state
.. automethod:: culebra.trainer.aco.abc.ACOFS2D._save_state
.. automethod:: culebra.trainer.aco.abc.ACOFS2D._set_state
.. automethod:: culebra.trainer.aco.abc.ACOFS2D._start_iteration
.. automethod:: culebra.trainer.aco.abc.ACOFS2D._termination_criterion
.. automethod:: culebra.trainer.aco.abc.ACOFS2D._unfeasible_nodes
.. automethod:: culebra.trainer.aco.abc.ACOFS2D._update_logbook
.. automethod:: culebra.trainer.aco.abc.ACOFS2D._update_pheromone