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

:py:class:`wrapper.single_pop.NSGA` class
=========================================

.. autoclass:: wrapper.single_pop.NSGA

Class attributes
----------------
.. autoattribute:: wrapper.single_pop.NSGA.stats_names
.. autoattribute:: wrapper.single_pop.NSGA.objective_stats

Properties
----------
.. autoproperty:: wrapper.single_pop.NSGA.individual_cls
.. autoproperty:: wrapper.single_pop.NSGA.species
.. autoproperty:: wrapper.single_pop.NSGA.fitness_function
.. autoproperty:: wrapper.single_pop.NSGA.num_gens
.. autoproperty:: wrapper.single_pop.NSGA.pop_size
.. autoproperty:: wrapper.single_pop.NSGA.crossover_func
.. autoproperty:: wrapper.single_pop.NSGA.mutation_func
.. autoproperty:: wrapper.single_pop.NSGA.selection_func
.. autoproperty:: wrapper.single_pop.NSGA.crossover_prob
.. autoproperty:: wrapper.single_pop.NSGA.mutation_prob
.. autoproperty:: wrapper.single_pop.NSGA.gene_ind_mutation_prob
.. autoproperty:: wrapper.single_pop.NSGA.selection_func_params
.. autoproperty:: wrapper.single_pop.NSGA.nsga3_reference_points_p
.. autoproperty:: wrapper.single_pop.NSGA.nsga3_reference_points_scaling
.. autoproperty:: wrapper.single_pop.NSGA.checkpoint_enable
.. autoproperty:: wrapper.single_pop.NSGA.checkpoint_freq
.. autoproperty:: wrapper.single_pop.NSGA.checkpoint_filename
.. autoproperty:: wrapper.single_pop.NSGA.verbose
.. autoproperty:: wrapper.single_pop.NSGA.random_seed
.. autoproperty:: wrapper.single_pop.NSGA.pop
.. autoproperty:: wrapper.single_pop.NSGA.nsga3_reference_points
.. autoproperty:: wrapper.single_pop.NSGA.logbook
.. autoproperty:: wrapper.single_pop.NSGA.num_evals
.. autoproperty:: wrapper.single_pop.NSGA.runtime
.. autoproperty:: wrapper.single_pop.NSGA.index
.. autoproperty:: wrapper.single_pop.NSGA.container
.. autoproperty:: wrapper.single_pop.NSGA.representatives

Private Properties
------------------
.. autoproperty:: wrapper.single_pop.NSGA._state

Methods
-------
.. automethod:: wrapper.single_pop.NSGA.reset
.. automethod:: wrapper.single_pop.NSGA.evaluate
.. automethod:: wrapper.single_pop.NSGA.best_solutions
.. automethod:: wrapper.single_pop.NSGA.best_representatives
.. automethod:: wrapper.single_pop.NSGA.train
.. automethod:: wrapper.single_pop.NSGA.test

Private methods
---------------
.. automethod:: wrapper.single_pop.NSGA._save_state
.. automethod:: wrapper.single_pop.NSGA._load_state
.. automethod:: wrapper.single_pop.NSGA._generate_initial_pop
.. automethod:: wrapper.single_pop.NSGA._evaluate_pop
.. automethod:: wrapper.single_pop.NSGA._do_generation_stats
.. automethod:: wrapper.single_pop.NSGA._new_state
.. automethod:: wrapper.single_pop.NSGA._reset_state
.. automethod:: wrapper.single_pop.NSGA._init_state
.. automethod:: wrapper.single_pop.NSGA._init_internals
.. automethod:: wrapper.single_pop.NSGA._reset_internals
.. automethod:: wrapper.single_pop.NSGA._init_search
.. automethod:: wrapper.single_pop.NSGA._start_generation
.. automethod:: wrapper.single_pop.NSGA._preprocess_generation
.. automethod:: wrapper.single_pop.NSGA._do_generation
.. automethod:: wrapper.single_pop.NSGA._postprocess_generation
.. automethod:: wrapper.single_pop.NSGA._finish_generation
.. automethod:: wrapper.single_pop.NSGA._search
.. automethod:: wrapper.single_pop.NSGA._finish_search
.. automethod:: wrapper.single_pop.NSGA._init_representatives
