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

:class:`culebra.trainer.ea.EA` class
====================================

.. autoclass:: culebra.trainer.ea.EA

Class methods
-------------
.. automethod:: culebra.trainer.ea.EA.load

Properties
----------
.. autoproperty:: culebra.trainer.ea.EA.checkpoint_activation
.. autoproperty:: culebra.trainer.ea.EA.checkpoint_basename
.. autoproperty:: culebra.trainer.ea.EA.checkpoint_filename
.. autoproperty:: culebra.trainer.ea.EA.checkpoint_freq
.. autoproperty:: culebra.trainer.ea.EA.container
.. autoproperty:: culebra.trainer.ea.EA.cooperative_fitness_estimation_func
.. autoproperty:: culebra.trainer.ea.EA.cooperators
.. autoproperty:: culebra.trainer.ea.EA.crossover_func
.. autoproperty:: culebra.trainer.ea.EA.crossover_prob
.. autoproperty:: culebra.trainer.ea.EA.current_iter
.. autoproperty:: culebra.trainer.ea.EA.custom_termination_func
.. autoproperty:: culebra.trainer.ea.EA.fitness_func
.. autoproperty:: culebra.trainer.ea.EA.gene_ind_mutation_prob
.. autoproperty:: culebra.trainer.ea.EA.index
.. autoproperty:: culebra.trainer.ea.EA.iteration_metric_names
.. autoproperty:: culebra.trainer.ea.EA.iteration_obj_stats
.. autoproperty:: culebra.trainer.ea.EA.logbook
.. autoproperty:: culebra.trainer.ea.EA.max_num_iters
.. autoproperty:: culebra.trainer.ea.EA.mutation_func
.. autoproperty:: culebra.trainer.ea.EA.mutation_prob
.. autoproperty:: culebra.trainer.ea.EA.num_evals
.. autoproperty:: culebra.trainer.ea.EA.num_iters
.. autoproperty:: culebra.trainer.ea.EA.pop
.. autoproperty:: culebra.trainer.ea.EA.pop_size
.. autoproperty:: culebra.trainer.ea.EA.random_seed
.. autoproperty:: culebra.trainer.ea.EA.receive_representatives_func
.. autoproperty:: culebra.trainer.ea.EA.runtime
.. autoproperty:: culebra.trainer.ea.EA.send_representatives_func
.. autoproperty:: culebra.trainer.ea.EA.selection_func
.. autoproperty:: culebra.trainer.ea.EA.solution_cls
.. autoproperty:: culebra.trainer.ea.EA.species
.. autoproperty:: culebra.trainer.ea.EA.state_proxy
.. autoproperty:: culebra.trainer.ea.EA.training_finished
.. autoproperty:: culebra.trainer.ea.EA.verbosity

Private properties
------------------
.. autoproperty:: culebra.trainer.ea.EA._default_checkpoint_activation
.. autoproperty:: culebra.trainer.ea.EA._default_checkpoint_basename
.. autoproperty:: culebra.trainer.ea.EA._default_checkpoint_freq
.. autoproperty:: culebra.trainer.ea.EA._default_cooperative_fitness_estimation_func
.. autoproperty:: culebra.trainer.ea.EA._default_crossover_func
.. autoproperty:: culebra.trainer.ea.EA._default_crossover_prob
.. autoproperty:: culebra.trainer.ea.EA._default_gene_ind_mutation_prob
.. autoproperty:: culebra.trainer.ea.EA._default_index
.. autoproperty:: culebra.trainer.ea.EA._default_max_num_iters
.. autoproperty:: culebra.trainer.ea.EA._default_mutation_func
.. autoproperty:: culebra.trainer.ea.EA._default_mutation_prob
.. autoproperty:: culebra.trainer.ea.EA._default_pop_size
.. autoproperty:: culebra.trainer.ea.EA._default_receive_representatives_func
.. autoproperty:: culebra.trainer.ea.EA._default_selection_func
.. autoproperty:: culebra.trainer.ea.EA._default_send_representatives_func
.. autoproperty:: culebra.trainer.ea.EA._default_verbosity

Methods
-------
.. automethod:: culebra.trainer.ea.EA.best_cooperators
.. automethod:: culebra.trainer.ea.EA.best_solutions
.. automethod:: culebra.trainer.ea.EA.dump
.. automethod:: culebra.trainer.ea.EA.evaluate
.. automethod:: culebra.trainer.ea.EA.integrate_representatives
.. automethod:: culebra.trainer.ea.EA.reset
.. automethod:: culebra.trainer.ea.EA.select_representatives
.. automethod:: culebra.trainer.ea.EA.test
.. automethod:: culebra.trainer.ea.EA.train

Private methods
---------------
.. automethod:: culebra.trainer.ea.EA._default_termination_func
.. automethod:: culebra.trainer.ea.EA._do_iteration
.. automethod:: culebra.trainer.ea.EA._do_training
.. automethod:: culebra.trainer.ea.EA._evaluate_several
.. automethod:: culebra.trainer.ea.EA._finish_iteration
.. automethod:: culebra.trainer.ea.EA._finish_training
.. automethod:: culebra.trainer.ea.EA._generate_cooperators
.. automethod:: culebra.trainer.ea.EA._generate_pop
.. automethod:: culebra.trainer.ea.EA._get_iteration_metrics
.. automethod:: culebra.trainer.ea.EA._get_objective_stats
.. automethod:: culebra.trainer.ea.EA._get_state
.. automethod:: culebra.trainer.ea.EA._init_internals
.. automethod:: culebra.trainer.ea.EA._init_state
.. automethod:: culebra.trainer.ea.EA._init_training
.. automethod:: culebra.trainer.ea.EA._load_state
.. automethod:: culebra.trainer.ea.EA._new_state
.. automethod:: culebra.trainer.ea.EA._reset_internals
.. automethod:: culebra.trainer.ea.EA._reset_state
.. automethod:: culebra.trainer.ea.EA._save_state
.. automethod:: culebra.trainer.ea.EA._set_state
.. automethod:: culebra.trainer.ea.EA._start_iteration
.. automethod:: culebra.trainer.ea.EA._termination_criterion
.. automethod:: culebra.trainer.ea.EA._update_logbook
