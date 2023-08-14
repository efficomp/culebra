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

:py:class:`culebra.trainer.ea.abc.IslandsEA` class
==================================================

.. autoclass:: culebra.trainer.ea.abc.IslandsEA

Class attributes
----------------
.. autoattribute:: culebra.trainer.ea.abc.IslandsEA.stats_names
.. autoattribute:: culebra.trainer.ea.abc.IslandsEA.objective_stats

Properties
----------
.. autoproperty:: culebra.trainer.ea.abc.IslandsEA.solution_cls
.. autoproperty:: culebra.trainer.ea.abc.IslandsEA.species
.. autoproperty:: culebra.trainer.ea.abc.IslandsEA.fitness_function
.. autoproperty:: culebra.trainer.ea.abc.IslandsEA.subpop_trainer_cls
.. autoproperty:: culebra.trainer.ea.abc.IslandsEA.max_num_iters
.. autoproperty:: culebra.trainer.ea.abc.IslandsEA.current_iter
.. autoproperty:: culebra.trainer.ea.abc.IslandsEA.custom_termination_func
.. autoproperty:: culebra.trainer.ea.abc.IslandsEA.num_subpops
.. autoproperty:: culebra.trainer.ea.abc.IslandsEA.representation_size
.. autoproperty:: culebra.trainer.ea.abc.IslandsEA.representation_freq
.. autoproperty:: culebra.trainer.ea.abc.IslandsEA.representation_topology_func
.. autoproperty:: culebra.trainer.ea.abc.IslandsEA.representation_topology_func_params
.. autoproperty:: culebra.trainer.ea.abc.IslandsEA.representation_selection_func
.. autoproperty:: culebra.trainer.ea.abc.IslandsEA.representation_selection_func_params
.. autoproperty:: culebra.trainer.ea.abc.IslandsEA.checkpoint_enable
.. autoproperty:: culebra.trainer.ea.abc.IslandsEA.checkpoint_freq
.. autoproperty:: culebra.trainer.ea.abc.IslandsEA.checkpoint_filename
.. autoproperty:: culebra.trainer.ea.abc.IslandsEA.verbose
.. autoproperty:: culebra.trainer.ea.abc.IslandsEA.random_seed
.. autoproperty:: culebra.trainer.ea.abc.IslandsEA.logbook
.. autoproperty:: culebra.trainer.ea.abc.IslandsEA.num_evals
.. autoproperty:: culebra.trainer.ea.abc.IslandsEA.runtime
.. autoproperty:: culebra.trainer.ea.abc.IslandsEA.index
.. autoproperty:: culebra.trainer.ea.abc.IslandsEA.container
.. autoproperty:: culebra.trainer.ea.abc.IslandsEA.representatives
.. autoproperty:: culebra.trainer.ea.abc.IslandsEA.subpop_trainer_params
.. autoproperty:: culebra.trainer.ea.abc.IslandsEA.subpop_trainer_checkpoint_filenames
.. autoproperty:: culebra.trainer.ea.abc.IslandsEA.subpop_trainers


Private properties
------------------
.. autoproperty:: culebra.trainer.ea.abc.IslandsEA._state
.. autoproperty:: culebra.trainer.ea.abc.IslandsEA._subpop_suffixes

Static methods
--------------
.. automethod:: culebra.trainer.ea.abc.IslandsEA.receive_representatives
.. automethod:: culebra.trainer.ea.abc.IslandsEA.send_representatives

Methods
-------
.. automethod:: culebra.trainer.ea.abc.IslandsEA.reset
.. automethod:: culebra.trainer.ea.abc.IslandsEA.evaluate
.. automethod:: culebra.trainer.ea.abc.IslandsEA.best_solutions
.. automethod:: culebra.trainer.ea.abc.IslandsEA.best_representatives
.. automethod:: culebra.trainer.ea.abc.IslandsEA.train
.. automethod:: culebra.trainer.ea.abc.IslandsEA.test

Private methods
---------------
.. automethod:: culebra.trainer.ea.abc.IslandsEA._generate_subpop_trainers
.. automethod:: culebra.trainer.ea.abc.IslandsEA._save_state
.. automethod:: culebra.trainer.ea.abc.IslandsEA._load_state
.. automethod:: culebra.trainer.ea.abc.IslandsEA._new_state
.. automethod:: culebra.trainer.ea.abc.IslandsEA._init_state
.. automethod:: culebra.trainer.ea.abc.IslandsEA._reset_state
.. automethod:: culebra.trainer.ea.abc.IslandsEA._init_internals
.. automethod:: culebra.trainer.ea.abc.IslandsEA._reset_internals
.. automethod:: culebra.trainer.ea.abc.IslandsEA._init_search
.. automethod:: culebra.trainer.ea.abc.IslandsEA._search
.. automethod:: culebra.trainer.ea.abc.IslandsEA._finish_search
.. automethod:: culebra.trainer.ea.abc.IslandsEA._start_iteration
.. automethod:: culebra.trainer.ea.abc.IslandsEA._preprocess_iteration
.. automethod:: culebra.trainer.ea.abc.IslandsEA._do_iteration
.. automethod:: culebra.trainer.ea.abc.IslandsEA._postprocess_iteration
.. automethod:: culebra.trainer.ea.abc.IslandsEA._finish_iteration
.. automethod:: culebra.trainer.ea.abc.IslandsEA._do_iteration_stats
.. automethod:: culebra.trainer.ea.abc.IslandsEA._default_termination_func
.. automethod:: culebra.trainer.ea.abc.IslandsEA._termination_criterion
.. automethod:: culebra.trainer.ea.abc.IslandsEA._init_representatives
