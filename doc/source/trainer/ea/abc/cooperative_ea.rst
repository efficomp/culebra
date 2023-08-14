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

:py:class:`culebra.trainer.ea.abc.CooperativeEA` class
======================================================

.. autoclass:: culebra.trainer.ea.abc.CooperativeEA

Class attributes
----------------
.. autoattribute:: culebra.trainer.ea.abc.CooperativeEA.stats_names
.. autoattribute:: culebra.trainer.ea.abc.CooperativeEA.objective_stats

Properties
----------
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.solution_classes
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.species
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.fitness_function
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.subpop_trainer_cls
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.max_num_iters
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.current_iter
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.custom_termination_func
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.pop_sizes
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.crossover_funcs
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.mutation_funcs
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.selection_funcs
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.crossover_probs
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.mutation_probs
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.gene_ind_mutation_probs
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.selection_funcs_params
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.num_subpops
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.representation_size
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.representation_freq
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.representation_topology_func
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.representation_topology_func_params
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.representation_selection_func
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.representation_selection_func_params
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.checkpoint_enable
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.checkpoint_freq
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.checkpoint_filename
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.verbose
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.random_seed
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.logbook
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.num_evals
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.runtime
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.index
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.container
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.representatives
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.subpop_trainer_params
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.subpop_trainer_checkpoint_filenames
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA.subpop_trainers


Private properties
------------------
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA._state
.. autoproperty:: culebra.trainer.ea.abc.CooperativeEA._subpop_suffixes

Static methods
--------------
.. automethod:: culebra.trainer.ea.abc.CooperativeEA.receive_representatives
.. automethod:: culebra.trainer.ea.abc.CooperativeEA.send_representatives

Private static methods
----------------------
.. automethod:: culebra.trainer.ea.abc.CooperativeEA._init_subpop_trainer_representatives

Methods
-------
.. automethod:: culebra.trainer.ea.abc.CooperativeEA.reset
.. automethod:: culebra.trainer.ea.abc.CooperativeEA.evaluate
.. automethod:: culebra.trainer.ea.abc.CooperativeEA.best_solutions
.. automethod:: culebra.trainer.ea.abc.CooperativeEA.best_representatives
.. automethod:: culebra.trainer.ea.abc.CooperativeEA.train
.. automethod:: culebra.trainer.ea.abc.CooperativeEA.test

Private methods
---------------
.. automethod:: culebra.trainer.ea.abc.CooperativeEA._generate_subpop_trainers
.. automethod:: culebra.trainer.ea.abc.CooperativeEA._save_state
.. automethod:: culebra.trainer.ea.abc.CooperativeEA._load_state
.. automethod:: culebra.trainer.ea.abc.CooperativeEA._new_state
.. automethod:: culebra.trainer.ea.abc.CooperativeEA._init_state
.. automethod:: culebra.trainer.ea.abc.CooperativeEA._reset_state
.. automethod:: culebra.trainer.ea.abc.CooperativeEA._init_internals
.. automethod:: culebra.trainer.ea.abc.CooperativeEA._reset_internals
.. automethod:: culebra.trainer.ea.abc.CooperativeEA._init_search
.. automethod:: culebra.trainer.ea.abc.CooperativeEA._search
.. automethod:: culebra.trainer.ea.abc.CooperativeEA._finish_search
.. automethod:: culebra.trainer.ea.abc.CooperativeEA._start_iteration
.. automethod:: culebra.trainer.ea.abc.CooperativeEA._preprocess_iteration
.. automethod:: culebra.trainer.ea.abc.CooperativeEA._do_iteration
.. automethod:: culebra.trainer.ea.abc.CooperativeEA._postprocess_iteration
.. automethod:: culebra.trainer.ea.abc.CooperativeEA._finish_iteration
.. automethod:: culebra.trainer.ea.abc.CooperativeEA._do_iteration_stats
.. automethod:: culebra.trainer.ea.abc.CooperativeEA._default_termination_func
.. automethod:: culebra.trainer.ea.abc.CooperativeEA._termination_criterion
.. automethod:: culebra.trainer.ea.abc.CooperativeEA._init_representatives
