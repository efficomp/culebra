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

:class:`culebra.trainer.abc.CentralizedTrainer` class
=====================================================

.. autoclass:: culebra.trainer.abc.CentralizedTrainer

Class methods
-------------
.. automethod:: culebra.trainer.abc.CentralizedTrainer.load

Properties
----------
.. autoproperty:: culebra.trainer.abc.CentralizedTrainer.checkpoint_activation
.. autoproperty:: culebra.trainer.abc.CentralizedTrainer.checkpoint_basename
.. autoproperty:: culebra.trainer.abc.CentralizedTrainer.checkpoint_filename
.. autoproperty:: culebra.trainer.abc.CentralizedTrainer.checkpoint_freq
.. autoproperty:: culebra.trainer.abc.CentralizedTrainer.container
.. autoproperty:: culebra.trainer.abc.CentralizedTrainer.cooperative_fitness_estimation_func
.. autoproperty:: culebra.trainer.abc.CentralizedTrainer.cooperators
.. autoproperty:: culebra.trainer.abc.CentralizedTrainer.current_iter
.. autoproperty:: culebra.trainer.abc.CentralizedTrainer.custom_termination_func
.. autoproperty:: culebra.trainer.abc.CentralizedTrainer.fitness_func
.. autoproperty:: culebra.trainer.abc.CentralizedTrainer.index
.. autoproperty:: culebra.trainer.abc.CentralizedTrainer.iteration_metric_names
.. autoproperty:: culebra.trainer.abc.CentralizedTrainer.iteration_obj_stats
.. autoproperty:: culebra.trainer.abc.CentralizedTrainer.logbook
.. autoproperty:: culebra.trainer.abc.CentralizedTrainer.max_num_iters
.. autoproperty:: culebra.trainer.abc.CentralizedTrainer.num_evals
.. autoproperty:: culebra.trainer.abc.CentralizedTrainer.num_iters
.. autoproperty:: culebra.trainer.abc.CentralizedTrainer.random_seed
.. autoproperty:: culebra.trainer.abc.CentralizedTrainer.receive_representatives_func
.. autoproperty:: culebra.trainer.abc.CentralizedTrainer.runtime
.. autoproperty:: culebra.trainer.abc.CentralizedTrainer.send_representatives_func
.. autoproperty:: culebra.trainer.abc.CentralizedTrainer.solution_cls
.. autoproperty:: culebra.trainer.abc.CentralizedTrainer.species
.. autoproperty:: culebra.trainer.abc.CentralizedTrainer.state_proxy
.. autoproperty:: culebra.trainer.abc.CentralizedTrainer.training_finished
.. autoproperty:: culebra.trainer.abc.CentralizedTrainer.verbosity

Private properties
------------------
.. autoproperty:: culebra.trainer.abc.CentralizedTrainer._default_checkpoint_activation
.. autoproperty:: culebra.trainer.abc.CentralizedTrainer._default_checkpoint_basename
.. autoproperty:: culebra.trainer.abc.CentralizedTrainer._default_checkpoint_freq
.. autoproperty:: culebra.trainer.abc.CentralizedTrainer._default_cooperative_fitness_estimation_func
.. autoproperty:: culebra.trainer.abc.CentralizedTrainer._default_index
.. autoproperty:: culebra.trainer.abc.CentralizedTrainer._default_max_num_iters
.. autoproperty:: culebra.trainer.abc.CentralizedTrainer._default_receive_representatives_func
.. autoproperty:: culebra.trainer.abc.CentralizedTrainer._default_send_representatives_func
.. autoproperty:: culebra.trainer.abc.CentralizedTrainer._default_verbosity

Methods
-------
.. automethod:: culebra.trainer.abc.CentralizedTrainer.best_cooperators
.. automethod:: culebra.trainer.abc.CentralizedTrainer.best_solutions
.. automethod:: culebra.trainer.abc.CentralizedTrainer.dump
.. automethod:: culebra.trainer.abc.CentralizedTrainer.evaluate
.. automethod:: culebra.trainer.abc.CentralizedTrainer.integrate_representatives
.. automethod:: culebra.trainer.abc.CentralizedTrainer.reset
.. automethod:: culebra.trainer.abc.CentralizedTrainer.select_representatives
.. automethod:: culebra.trainer.abc.CentralizedTrainer.test
.. automethod:: culebra.trainer.abc.CentralizedTrainer.train

Private methods
---------------
.. automethod:: culebra.trainer.abc.CentralizedTrainer._default_termination_func
.. automethod:: culebra.trainer.abc.CentralizedTrainer._do_iteration
.. automethod:: culebra.trainer.abc.CentralizedTrainer._do_training
.. automethod:: culebra.trainer.abc.CentralizedTrainer._finish_iteration
.. automethod:: culebra.trainer.abc.CentralizedTrainer._finish_training
.. automethod:: culebra.trainer.abc.CentralizedTrainer._generate_cooperators
.. automethod:: culebra.trainer.abc.CentralizedTrainer._get_iteration_metrics
.. automethod:: culebra.trainer.abc.CentralizedTrainer._get_objective_stats
.. automethod:: culebra.trainer.abc.CentralizedTrainer._get_state
.. automethod:: culebra.trainer.abc.CentralizedTrainer._init_internals
.. automethod:: culebra.trainer.abc.CentralizedTrainer._init_state
.. automethod:: culebra.trainer.abc.CentralizedTrainer._init_training
.. automethod:: culebra.trainer.abc.CentralizedTrainer._load_state
.. automethod:: culebra.trainer.abc.CentralizedTrainer._new_state
.. automethod:: culebra.trainer.abc.CentralizedTrainer._reset_internals
.. automethod:: culebra.trainer.abc.CentralizedTrainer._reset_state
.. automethod:: culebra.trainer.abc.CentralizedTrainer._save_state
.. automethod:: culebra.trainer.abc.CentralizedTrainer._set_state
.. automethod:: culebra.trainer.abc.CentralizedTrainer._start_iteration
.. automethod:: culebra.trainer.abc.CentralizedTrainer._termination_criterion
.. automethod:: culebra.trainer.abc.CentralizedTrainer._update_logbook
