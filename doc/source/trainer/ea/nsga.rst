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

:class:`culebra.trainer.ea.NSGA` class
======================================

.. autoclass:: culebra.trainer.ea.NSGA

Class methods
-------------
.. automethod:: culebra.trainer.ea.NSGA.load

Properties
----------
.. autoproperty:: culebra.trainer.ea.NSGA.checkpoint_activation
.. autoproperty:: culebra.trainer.ea.NSGA.checkpoint_basename
.. autoproperty:: culebra.trainer.ea.NSGA.checkpoint_filename
.. autoproperty:: culebra.trainer.ea.NSGA.checkpoint_freq
.. autoproperty:: culebra.trainer.ea.NSGA.container
.. autoproperty:: culebra.trainer.ea.NSGA.cooperative_fitness_estimation_func
.. autoproperty:: culebra.trainer.ea.NSGA.cooperators
.. autoproperty:: culebra.trainer.ea.NSGA.crossover_func
.. autoproperty:: culebra.trainer.ea.NSGA.crossover_prob
.. autoproperty:: culebra.trainer.ea.NSGA.current_iter
.. autoproperty:: culebra.trainer.ea.NSGA.custom_termination_func
.. autoproperty:: culebra.trainer.ea.NSGA.fitness_func
.. autoproperty:: culebra.trainer.ea.NSGA.gene_ind_mutation_prob
.. autoproperty:: culebra.trainer.ea.NSGA.index
.. autoproperty:: culebra.trainer.ea.NSGA.iteration_metric_names
.. autoproperty:: culebra.trainer.ea.NSGA.iteration_obj_stats
.. autoproperty:: culebra.trainer.ea.NSGA.logbook
.. autoproperty:: culebra.trainer.ea.NSGA.max_num_iters
.. autoproperty:: culebra.trainer.ea.NSGA.mutation_func
.. autoproperty:: culebra.trainer.ea.NSGA.mutation_prob
.. autoproperty:: culebra.trainer.ea.NSGA.nsga3_reference_points
.. autoproperty:: culebra.trainer.ea.NSGA.nsga3_reference_points_p
.. autoproperty:: culebra.trainer.ea.NSGA.nsga3_reference_points_scaling
.. autoproperty:: culebra.trainer.ea.NSGA.num_evals
.. autoproperty:: culebra.trainer.ea.NSGA.num_iters
.. autoproperty:: culebra.trainer.ea.NSGA.pop
.. autoproperty:: culebra.trainer.ea.NSGA.pop_size
.. autoproperty:: culebra.trainer.ea.NSGA.random_seed
.. autoproperty:: culebra.trainer.ea.NSGA.receive_representatives_func
.. autoproperty:: culebra.trainer.ea.NSGA.runtime
.. autoproperty:: culebra.trainer.ea.NSGA.send_representatives_func
.. autoproperty:: culebra.trainer.ea.NSGA.selection_func
.. autoproperty:: culebra.trainer.ea.NSGA.solution_cls
.. autoproperty:: culebra.trainer.ea.NSGA.species
.. autoproperty:: culebra.trainer.ea.NSGA.state_proxy
.. autoproperty:: culebra.trainer.ea.NSGA.training_finished
.. autoproperty:: culebra.trainer.ea.NSGA.verbosity

Private properties
------------------
.. autoproperty:: culebra.trainer.ea.NSGA._default_checkpoint_activation
.. autoproperty:: culebra.trainer.ea.NSGA._default_checkpoint_basename
.. autoproperty:: culebra.trainer.ea.NSGA._default_checkpoint_freq
.. autoproperty:: culebra.trainer.ea.NSGA._default_cooperative_fitness_estimation_func
.. autoproperty:: culebra.trainer.ea.NSGA._default_crossover_func
.. autoproperty:: culebra.trainer.ea.NSGA._default_crossover_prob
.. autoproperty:: culebra.trainer.ea.NSGA._default_gene_ind_mutation_prob
.. autoproperty:: culebra.trainer.ea.NSGA._default_index
.. autoproperty:: culebra.trainer.ea.NSGA._default_max_num_iters
.. autoproperty:: culebra.trainer.ea.NSGA._default_mutation_func
.. autoproperty:: culebra.trainer.ea.NSGA._default_mutation_prob
.. autoproperty:: culebra.trainer.ea.NSGA._default_nsga3_reference_points_p
.. autoproperty:: culebra.trainer.ea.NSGA._default_pop_size
.. autoproperty:: culebra.trainer.ea.NSGA._default_receive_representatives_func
.. autoproperty:: culebra.trainer.ea.NSGA._default_selection_func
.. autoproperty:: culebra.trainer.ea.NSGA._default_send_representatives_func
.. autoproperty:: culebra.trainer.ea.NSGA._default_verbosity

Methods
-------
.. automethod:: culebra.trainer.ea.NSGA.best_cooperators
.. automethod:: culebra.trainer.ea.NSGA.best_solutions
.. automethod:: culebra.trainer.ea.NSGA.dump
.. automethod:: culebra.trainer.ea.NSGA.evaluate
.. automethod:: culebra.trainer.ea.NSGA.integrate_representatives
.. automethod:: culebra.trainer.ea.NSGA.reset
.. automethod:: culebra.trainer.ea.NSGA.select_representatives
.. automethod:: culebra.trainer.ea.NSGA.test
.. automethod:: culebra.trainer.ea.NSGA.train

Private methods
---------------
.. automethod:: culebra.trainer.ea.NSGA._default_termination_func
.. automethod:: culebra.trainer.ea.NSGA._do_iteration
.. automethod:: culebra.trainer.ea.NSGA._do_training
.. automethod:: culebra.trainer.ea.NSGA._evaluate_several
.. automethod:: culebra.trainer.ea.NSGA._finish_iteration
.. automethod:: culebra.trainer.ea.NSGA._finish_training
.. automethod:: culebra.trainer.ea.NSGA._generate_cooperators
.. automethod:: culebra.trainer.ea.NSGA._generate_pop
.. automethod:: culebra.trainer.ea.NSGA._get_iteration_metrics
.. automethod:: culebra.trainer.ea.NSGA._get_objective_stats
.. automethod:: culebra.trainer.ea.NSGA._get_state
.. automethod:: culebra.trainer.ea.NSGA._init_internals
.. automethod:: culebra.trainer.ea.NSGA._init_state
.. automethod:: culebra.trainer.ea.NSGA._init_training
.. automethod:: culebra.trainer.ea.NSGA._load_state
.. automethod:: culebra.trainer.ea.NSGA._new_state
.. automethod:: culebra.trainer.ea.NSGA._reset_internals
.. automethod:: culebra.trainer.ea.NSGA._reset_state
.. automethod:: culebra.trainer.ea.NSGA._save_state
.. automethod:: culebra.trainer.ea.NSGA._set_state
.. automethod:: culebra.trainer.ea.NSGA._start_iteration
.. automethod:: culebra.trainer.ea.NSGA._termination_criterion
.. automethod:: culebra.trainer.ea.NSGA._update_logbook
