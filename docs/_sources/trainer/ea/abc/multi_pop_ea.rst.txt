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

:class:`culebra.trainer.ea.abc.MultiPopEA` class
================================================

.. autoclass:: culebra.trainer.ea.abc.MultiPopEA

Class attributes
----------------
.. autoattribute:: culebra.trainer.ea.abc.MultiPopEA.objective_stats
.. autoattribute:: culebra.trainer.ea.abc.MultiPopEA.stats_names

Class methods
-------------
.. automethod:: culebra.trainer.ea.abc.MultiPopEA.load

Properties
----------
.. autoproperty:: culebra.trainer.ea.abc.MultiPopEA.checkpoint_activation
.. autoproperty:: culebra.trainer.ea.abc.MultiPopEA.checkpoint_filename
.. autoproperty:: culebra.trainer.ea.abc.MultiPopEA.checkpoint_freq
.. autoproperty:: culebra.trainer.ea.abc.MultiPopEA.container
.. autoproperty:: culebra.trainer.ea.abc.MultiPopEA.current_iter
.. autoproperty:: culebra.trainer.ea.abc.MultiPopEA.custom_termination_func
.. autoproperty:: culebra.trainer.ea.abc.MultiPopEA.fitness_function
.. autoproperty:: culebra.trainer.ea.abc.MultiPopEA.index
.. autoproperty:: culebra.trainer.ea.abc.MultiPopEA.logbook
.. autoproperty:: culebra.trainer.ea.abc.MultiPopEA.max_num_iters
.. autoproperty:: culebra.trainer.ea.abc.MultiPopEA.num_evals
.. autoproperty:: culebra.trainer.ea.abc.MultiPopEA.num_subtrainers
.. autoproperty:: culebra.trainer.ea.abc.MultiPopEA.random_seed
.. autoproperty:: culebra.trainer.ea.abc.MultiPopEA.representation_freq
.. autoproperty:: culebra.trainer.ea.abc.MultiPopEA.representation_selection_func
.. autoproperty:: culebra.trainer.ea.abc.MultiPopEA.representation_selection_func_params
.. autoproperty:: culebra.trainer.ea.abc.MultiPopEA.representation_size
.. autoproperty:: culebra.trainer.ea.abc.MultiPopEA.representation_topology_func
.. autoproperty:: culebra.trainer.ea.abc.MultiPopEA.representation_topology_func_params
.. autoproperty:: culebra.trainer.ea.abc.MultiPopEA.representatives
.. autoproperty:: culebra.trainer.ea.abc.MultiPopEA.runtime
.. autoproperty:: culebra.trainer.ea.abc.MultiPopEA.subtrainer_checkpoint_filenames
.. autoproperty:: culebra.trainer.ea.abc.MultiPopEA.subtrainer_cls
.. autoproperty:: culebra.trainer.ea.abc.MultiPopEA.subtrainer_params
.. autoproperty:: culebra.trainer.ea.abc.MultiPopEA.subtrainers
.. autoproperty:: culebra.trainer.ea.abc.MultiPopEA.verbosity

Private properties
------------------
.. autoproperty:: culebra.trainer.ea.abc.MultiPopEA._default_checkpoint_activation
.. autoproperty:: culebra.trainer.ea.abc.MultiPopEA._default_checkpoint_filename
.. autoproperty:: culebra.trainer.ea.abc.MultiPopEA._default_checkpoint_freq
.. autoproperty:: culebra.trainer.ea.abc.MultiPopEA._default_index
.. autoproperty:: culebra.trainer.ea.abc.MultiPopEA._default_max_num_iters
.. autoproperty:: culebra.trainer.ea.abc.MultiPopEA._default_num_subtrainers
.. autoproperty:: culebra.trainer.ea.abc.MultiPopEA._default_representation_freq
.. autoproperty:: culebra.trainer.ea.abc.MultiPopEA._default_representation_selection_func
.. autoproperty:: culebra.trainer.ea.abc.MultiPopEA._default_representation_selection_func_params
.. autoproperty:: culebra.trainer.ea.abc.MultiPopEA._default_representation_size
.. autoproperty:: culebra.trainer.ea.abc.MultiPopEA._default_representation_topology_func
.. autoproperty:: culebra.trainer.ea.abc.MultiPopEA._default_representation_topology_func_params
.. autoproperty:: culebra.trainer.ea.abc.MultiPopEA._default_verbosity
.. autoproperty:: culebra.trainer.ea.abc.MultiPopEA._subtrainer_suffixes

Static methods
--------------
.. automethod:: culebra.trainer.ea.abc.MultiPopEA.receive_representatives
.. automethod:: culebra.trainer.ea.abc.MultiPopEA.send_representatives

Methods
-------
.. automethod:: culebra.trainer.ea.abc.MultiPopEA.best_representatives
.. automethod:: culebra.trainer.ea.abc.MultiPopEA.best_solutions
.. automethod:: culebra.trainer.ea.abc.MultiPopEA.dump
.. automethod:: culebra.trainer.ea.abc.MultiPopEA.evaluate
.. automethod:: culebra.trainer.ea.abc.MultiPopEA.reset
.. automethod:: culebra.trainer.ea.abc.MultiPopEA.test
.. automethod:: culebra.trainer.ea.abc.MultiPopEA.train

Private methods
---------------
.. automethod:: culebra.trainer.ea.abc.MultiPopEA._default_termination_func
.. automethod:: culebra.trainer.ea.abc.MultiPopEA._do_iteration
.. automethod:: culebra.trainer.ea.abc.MultiPopEA._do_iteration_stats
.. automethod:: culebra.trainer.ea.abc.MultiPopEA._finish_iteration
.. automethod:: culebra.trainer.ea.abc.MultiPopEA._finish_search
.. automethod:: culebra.trainer.ea.abc.MultiPopEA._generate_subtrainers
.. automethod:: culebra.trainer.ea.abc.MultiPopEA._get_state
.. automethod:: culebra.trainer.ea.abc.MultiPopEA._init_internals
.. automethod:: culebra.trainer.ea.abc.MultiPopEA._init_representatives
.. automethod:: culebra.trainer.ea.abc.MultiPopEA._init_search
.. automethod:: culebra.trainer.ea.abc.MultiPopEA._init_state
.. automethod:: culebra.trainer.ea.abc.MultiPopEA._load_state
.. automethod:: culebra.trainer.ea.abc.MultiPopEA._new_state
.. automethod:: culebra.trainer.ea.abc.MultiPopEA._postprocess_iteration
.. automethod:: culebra.trainer.ea.abc.MultiPopEA._preprocess_iteration
.. automethod:: culebra.trainer.ea.abc.MultiPopEA._reset_internals
.. automethod:: culebra.trainer.ea.abc.MultiPopEA._reset_state
.. automethod:: culebra.trainer.ea.abc.MultiPopEA._save_state
.. automethod:: culebra.trainer.ea.abc.MultiPopEA._search
.. automethod:: culebra.trainer.ea.abc.MultiPopEA._set_cooperative_fitness
.. automethod:: culebra.trainer.ea.abc.MultiPopEA._set_state
.. automethod:: culebra.trainer.ea.abc.MultiPopEA._start_iteration
.. automethod:: culebra.trainer.ea.abc.MultiPopEA._termination_criterion
