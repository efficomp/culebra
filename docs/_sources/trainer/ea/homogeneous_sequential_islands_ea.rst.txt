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

:py:class:`culebra.trainer.ea.HomogeneousSequentialIslandsEA` class
===================================================================

.. autoclass:: culebra.trainer.ea.HomogeneousSequentialIslandsEA

Class attributes
----------------
.. autoattribute:: culebra.trainer.ea.HomogeneousSequentialIslandsEA.stats_names
.. autoattribute:: culebra.trainer.ea.HomogeneousSequentialIslandsEA.objective_stats

Properties
----------
.. autoproperty:: culebra.trainer.ea.HomogeneousSequentialIslandsEA.solution_cls
.. autoproperty:: culebra.trainer.ea.HomogeneousSequentialIslandsEA.species
.. autoproperty:: culebra.trainer.ea.HomogeneousSequentialIslandsEA.fitness_function
.. autoproperty:: culebra.trainer.ea.HomogeneousSequentialIslandsEA.subpop_trainer_cls
.. autoproperty:: culebra.trainer.ea.HomogeneousSequentialIslandsEA.max_num_iters
.. autoproperty:: culebra.trainer.ea.HomogeneousSequentialIslandsEA.current_iter
.. autoproperty:: culebra.trainer.ea.HomogeneousSequentialIslandsEA.custom_termination_func
.. autoproperty:: culebra.trainer.ea.HomogeneousSequentialIslandsEA.pop_size
.. autoproperty:: culebra.trainer.ea.HomogeneousSequentialIslandsEA.crossover_func
.. autoproperty:: culebra.trainer.ea.HomogeneousSequentialIslandsEA.mutation_func
.. autoproperty:: culebra.trainer.ea.HomogeneousSequentialIslandsEA.selection_func
.. autoproperty:: culebra.trainer.ea.HomogeneousSequentialIslandsEA.crossover_prob
.. autoproperty:: culebra.trainer.ea.HomogeneousSequentialIslandsEA.mutation_prob
.. autoproperty:: culebra.trainer.ea.HomogeneousSequentialIslandsEA.gene_ind_mutation_prob
.. autoproperty:: culebra.trainer.ea.HomogeneousSequentialIslandsEA.selection_func_params
.. autoproperty:: culebra.trainer.ea.HomogeneousSequentialIslandsEA.num_subpops
.. autoproperty:: culebra.trainer.ea.HomogeneousSequentialIslandsEA.representation_size
.. autoproperty:: culebra.trainer.ea.HomogeneousSequentialIslandsEA.representation_freq
.. autoproperty:: culebra.trainer.ea.HomogeneousSequentialIslandsEA.representation_topology_func
.. autoproperty:: culebra.trainer.ea.HomogeneousSequentialIslandsEA.representation_topology_func_params
.. autoproperty:: culebra.trainer.ea.HomogeneousSequentialIslandsEA.representation_selection_func
.. autoproperty:: culebra.trainer.ea.HomogeneousSequentialIslandsEA.representation_selection_func_params
.. autoproperty:: culebra.trainer.ea.HomogeneousSequentialIslandsEA.checkpoint_enable
.. autoproperty:: culebra.trainer.ea.HomogeneousSequentialIslandsEA.checkpoint_freq
.. autoproperty:: culebra.trainer.ea.HomogeneousSequentialIslandsEA.checkpoint_filename
.. autoproperty:: culebra.trainer.ea.HomogeneousSequentialIslandsEA.verbose
.. autoproperty:: culebra.trainer.ea.HomogeneousSequentialIslandsEA.random_seed
.. autoproperty:: culebra.trainer.ea.HomogeneousSequentialIslandsEA.logbook
.. autoproperty:: culebra.trainer.ea.HomogeneousSequentialIslandsEA.num_evals
.. autoproperty:: culebra.trainer.ea.HomogeneousSequentialIslandsEA.runtime
.. autoproperty:: culebra.trainer.ea.HomogeneousSequentialIslandsEA.index
.. autoproperty:: culebra.trainer.ea.HomogeneousSequentialIslandsEA.container
.. autoproperty:: culebra.trainer.ea.HomogeneousSequentialIslandsEA.representatives
.. autoproperty:: culebra.trainer.ea.HomogeneousSequentialIslandsEA.subpop_trainer_params
.. autoproperty:: culebra.trainer.ea.HomogeneousSequentialIslandsEA.subpop_trainer_checkpoint_filenames
.. autoproperty:: culebra.trainer.ea.HomogeneousSequentialIslandsEA.subpop_trainers


Private properties
------------------
.. autoproperty:: culebra.trainer.ea.HomogeneousSequentialIslandsEA._state
.. autoproperty:: culebra.trainer.ea.HomogeneousSequentialIslandsEA._subpop_suffixes

Static methods
--------------
.. automethod:: culebra.trainer.ea.HomogeneousSequentialIslandsEA.receive_representatives
.. automethod:: culebra.trainer.ea.HomogeneousSequentialIslandsEA.send_representatives

Methods
-------
.. automethod:: culebra.trainer.ea.HomogeneousSequentialIslandsEA.reset
.. automethod:: culebra.trainer.ea.HomogeneousSequentialIslandsEA.evaluate
.. automethod:: culebra.trainer.ea.HomogeneousSequentialIslandsEA.best_solutions
.. automethod:: culebra.trainer.ea.HomogeneousSequentialIslandsEA.best_representatives
.. automethod:: culebra.trainer.ea.HomogeneousSequentialIslandsEA.train
.. automethod:: culebra.trainer.ea.HomogeneousSequentialIslandsEA.test

Private methods
---------------
.. automethod:: culebra.trainer.ea.HomogeneousSequentialIslandsEA._generate_subpop_trainers
.. automethod:: culebra.trainer.ea.HomogeneousSequentialIslandsEA._save_state
.. automethod:: culebra.trainer.ea.HomogeneousSequentialIslandsEA._load_state
.. automethod:: culebra.trainer.ea.HomogeneousSequentialIslandsEA._new_state
.. automethod:: culebra.trainer.ea.HomogeneousSequentialIslandsEA._init_state
.. automethod:: culebra.trainer.ea.HomogeneousSequentialIslandsEA._reset_state
.. automethod:: culebra.trainer.ea.HomogeneousSequentialIslandsEA._init_internals
.. automethod:: culebra.trainer.ea.HomogeneousSequentialIslandsEA._reset_internals
.. automethod:: culebra.trainer.ea.HomogeneousSequentialIslandsEA._init_search
.. automethod:: culebra.trainer.ea.HomogeneousSequentialIslandsEA._search
.. automethod:: culebra.trainer.ea.HomogeneousSequentialIslandsEA._finish_search
.. automethod:: culebra.trainer.ea.HomogeneousSequentialIslandsEA._start_iteration
.. automethod:: culebra.trainer.ea.HomogeneousSequentialIslandsEA._preprocess_iteration
.. automethod:: culebra.trainer.ea.HomogeneousSequentialIslandsEA._do_iteration
.. automethod:: culebra.trainer.ea.HomogeneousSequentialIslandsEA._postprocess_iteration
.. automethod:: culebra.trainer.ea.HomogeneousSequentialIslandsEA._finish_iteration
.. automethod:: culebra.trainer.ea.HomogeneousSequentialIslandsEA._do_iteration_stats
.. automethod:: culebra.trainer.ea.HomogeneousSequentialIslandsEA._default_termination_func
.. automethod:: culebra.trainer.ea.HomogeneousSequentialIslandsEA._termination_criterion
.. automethod:: culebra.trainer.ea.HomogeneousSequentialIslandsEA._init_representatives
