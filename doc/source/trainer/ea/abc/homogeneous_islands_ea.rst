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

:class:`culebra.trainer.ea.abc.HomogeneousIslandsEA` class
==========================================================

.. autoclass:: culebra.trainer.ea.abc.HomogeneousIslandsEA

Class attributes
----------------
.. autoattribute:: culebra.trainer.ea.abc.HomogeneousIslandsEA.objective_stats
.. autoattribute:: culebra.trainer.ea.abc.HomogeneousIslandsEA.stats_names

Class methods
-------------
.. automethod:: culebra.trainer.ea.abc.HomogeneousIslandsEA.load

Properties
----------
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousIslandsEA.checkpoint_enable
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousIslandsEA.checkpoint_filename
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousIslandsEA.checkpoint_freq
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousIslandsEA.container
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousIslandsEA.crossover_func
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousIslandsEA.crossover_prob
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousIslandsEA.current_iter
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousIslandsEA.custom_termination_func
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousIslandsEA.fitness_function
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousIslandsEA.gene_ind_mutation_prob
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousIslandsEA.index
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousIslandsEA.logbook
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousIslandsEA.max_num_iters
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousIslandsEA.mutation_func
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousIslandsEA.mutation_prob
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousIslandsEA.num_evals
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousIslandsEA.num_subtrainers
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousIslandsEA.pop_size
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousIslandsEA.random_seed
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousIslandsEA.representation_freq
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousIslandsEA.representation_selection_func
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousIslandsEA.representation_selection_func_params
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousIslandsEA.representation_size
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousIslandsEA.representation_topology_func
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousIslandsEA.representation_topology_func_params
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousIslandsEA.representatives
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousIslandsEA.runtime
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousIslandsEA.selection_func
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousIslandsEA.selection_func_params
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousIslandsEA.solution_cls
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousIslandsEA.species
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousIslandsEA.subtrainer_checkpoint_filenames
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousIslandsEA.subtrainer_cls
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousIslandsEA.subtrainer_params
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousIslandsEA.subtrainers
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousIslandsEA.verbose

Private properties
------------------
.. autoproperty:: culebra.trainer.ea.abc.HomogeneousIslandsEA._subtrainer_suffixes

Static methods
--------------
.. automethod:: culebra.trainer.ea.abc.HomogeneousIslandsEA.receive_representatives
.. automethod:: culebra.trainer.ea.abc.HomogeneousIslandsEA.send_representatives

Methods
-------
.. automethod:: culebra.trainer.ea.abc.HomogeneousIslandsEA.best_representatives
.. automethod:: culebra.trainer.ea.abc.HomogeneousIslandsEA.best_solutions
.. automethod:: culebra.trainer.ea.abc.HomogeneousIslandsEA.dump
.. automethod:: culebra.trainer.ea.abc.HomogeneousIslandsEA.evaluate
.. automethod:: culebra.trainer.ea.abc.HomogeneousIslandsEA.reset
.. automethod:: culebra.trainer.ea.abc.HomogeneousIslandsEA.test
.. automethod:: culebra.trainer.ea.abc.HomogeneousIslandsEA.train

Private methods
---------------
.. automethod:: culebra.trainer.ea.abc.HomogeneousIslandsEA._default_termination_func
.. automethod:: culebra.trainer.ea.abc.HomogeneousIslandsEA._do_iteration
.. automethod:: culebra.trainer.ea.abc.HomogeneousIslandsEA._do_iteration_stats
.. automethod:: culebra.trainer.ea.abc.HomogeneousIslandsEA._finish_iteration
.. automethod:: culebra.trainer.ea.abc.HomogeneousIslandsEA._finish_search
.. automethod:: culebra.trainer.ea.abc.HomogeneousIslandsEA._generate_subtrainers
.. automethod:: culebra.trainer.ea.abc.HomogeneousIslandsEA._get_state
.. automethod:: culebra.trainer.ea.abc.HomogeneousIslandsEA._init_internals
.. automethod:: culebra.trainer.ea.abc.HomogeneousIslandsEA._init_representatives
.. automethod:: culebra.trainer.ea.abc.HomogeneousIslandsEA._init_search
.. automethod:: culebra.trainer.ea.abc.HomogeneousIslandsEA._init_state
.. automethod:: culebra.trainer.ea.abc.HomogeneousIslandsEA._load_state
.. automethod:: culebra.trainer.ea.abc.HomogeneousIslandsEA._new_state
.. automethod:: culebra.trainer.ea.abc.HomogeneousIslandsEA._postprocess_iteration
.. automethod:: culebra.trainer.ea.abc.HomogeneousIslandsEA._preprocess_iteration
.. automethod:: culebra.trainer.ea.abc.HomogeneousIslandsEA._reset_internals
.. automethod:: culebra.trainer.ea.abc.HomogeneousIslandsEA._reset_state
.. automethod:: culebra.trainer.ea.abc.HomogeneousIslandsEA._save_state
.. automethod:: culebra.trainer.ea.abc.HomogeneousIslandsEA._search
.. automethod:: culebra.trainer.ea.abc.HomogeneousIslandsEA._set_cooperative_fitness
.. automethod:: culebra.trainer.ea.abc.HomogeneousIslandsEA._set_state
.. automethod:: culebra.trainer.ea.abc.HomogeneousIslandsEA._start_iteration
.. automethod:: culebra.trainer.ea.abc.HomogeneousIslandsEA._termination_criterion
