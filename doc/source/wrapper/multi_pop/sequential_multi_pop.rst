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

:py:class:`wrapper.multi_pop.SequentialMultiPop` class
======================================================

.. autoclass:: wrapper.multi_pop.SequentialMultiPop

Class attributes
----------------
.. autoattribute:: wrapper.multi_pop.SequentialMultiPop.stats_names
.. autoattribute:: wrapper.multi_pop.SequentialMultiPop.objective_stats

Properties
----------
.. autoproperty:: wrapper.multi_pop.SequentialMultiPop.fitness_function
.. autoproperty:: wrapper.multi_pop.SequentialMultiPop.subpop_wrapper_cls
.. autoproperty:: wrapper.multi_pop.SequentialMultiPop.subpop_wrapper_params
.. autoproperty:: wrapper.multi_pop.SequentialMultiPop.num_gens
.. autoproperty:: wrapper.multi_pop.SequentialMultiPop.num_subpops
.. autoproperty:: wrapper.multi_pop.SequentialMultiPop.representation_size
.. autoproperty:: wrapper.multi_pop.SequentialMultiPop.representation_freq
.. autoproperty:: wrapper.multi_pop.SequentialMultiPop.representation_topology_func
.. autoproperty:: wrapper.multi_pop.SequentialMultiPop.representation_topology_func_params
.. autoproperty:: wrapper.multi_pop.SequentialMultiPop.representation_selection_func
.. autoproperty:: wrapper.multi_pop.SequentialMultiPop.representation_selection_func_params
.. autoproperty:: wrapper.multi_pop.SequentialMultiPop.checkpoint_enable
.. autoproperty:: wrapper.multi_pop.SequentialMultiPop.checkpoint_freq
.. autoproperty:: wrapper.multi_pop.SequentialMultiPop.checkpoint_filename
.. autoproperty:: wrapper.multi_pop.SequentialMultiPop.verbose
.. autoproperty:: wrapper.multi_pop.SequentialMultiPop.random_seed
.. autoproperty:: wrapper.multi_pop.SequentialMultiPop.subpop_wrappers
.. autoproperty:: wrapper.multi_pop.SequentialMultiPop.subpop_wrapper_checkpoint_filenames
.. autoproperty:: wrapper.multi_pop.SequentialMultiPop.logbook
.. autoproperty:: wrapper.multi_pop.SequentialMultiPop.num_evals
.. autoproperty:: wrapper.multi_pop.SequentialMultiPop.runtime
.. autoproperty:: wrapper.multi_pop.SequentialMultiPop.index
.. autoproperty:: wrapper.multi_pop.SequentialMultiPop.container
.. autoproperty:: wrapper.multi_pop.SequentialMultiPop.representatives

Private Properties
------------------
.. autoproperty:: wrapper.multi_pop.SequentialMultiPop._state

Static methods
--------------
.. automethod:: wrapper.multi_pop.SequentialMultiPop.receive_representatives
.. automethod:: wrapper.multi_pop.SequentialMultiPop.send_representatives


Methods
-------
.. automethod:: wrapper.multi_pop.SequentialMultiPop.reset
.. automethod:: wrapper.multi_pop.SequentialMultiPop.evaluate
.. automethod:: wrapper.multi_pop.SequentialMultiPop.best_solutions
.. automethod:: wrapper.multi_pop.SequentialMultiPop.best_representatives
.. automethod:: wrapper.multi_pop.SequentialMultiPop.train
.. automethod:: wrapper.multi_pop.SequentialMultiPop.test


Private methods
---------------
.. automethod:: wrapper.multi_pop.SequentialMultiPop._generate_subpop_wrappers
.. automethod:: wrapper.multi_pop.SequentialMultiPop._save_state
.. automethod:: wrapper.multi_pop.SequentialMultiPop._load_state
.. automethod:: wrapper.multi_pop.SequentialMultiPop._new_state
.. automethod:: wrapper.multi_pop.SequentialMultiPop._reset_state
.. automethod:: wrapper.multi_pop.SequentialMultiPop._init_state
.. automethod:: wrapper.multi_pop.SequentialMultiPop._init_internals
.. automethod:: wrapper.multi_pop.SequentialMultiPop._reset_internals
.. automethod:: wrapper.multi_pop.SequentialMultiPop._init_search
.. automethod:: wrapper.multi_pop.SequentialMultiPop._start_generation
.. automethod:: wrapper.multi_pop.SequentialMultiPop._preprocess_generation
.. automethod:: wrapper.multi_pop.SequentialMultiPop._do_generation
.. automethod:: wrapper.multi_pop.SequentialMultiPop._postprocess_generation
.. automethod:: wrapper.multi_pop.SequentialMultiPop._finish_generation
.. automethod:: wrapper.multi_pop.SequentialMultiPop._search
.. automethod:: wrapper.multi_pop.SequentialMultiPop._finish_search
.. automethod:: wrapper.multi_pop.SequentialMultiPop._init_representatives
