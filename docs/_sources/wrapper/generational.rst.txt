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

:py:class:`wrapper.Generational` class
======================================

.. autoclass:: wrapper.Generational

Class attributes
----------------
.. autoattribute:: wrapper.Generational.stats_names
.. autoattribute:: wrapper.Generational.objective_stats

Properties
----------
.. autoproperty:: wrapper.Generational.fitness_function
.. autoproperty:: wrapper.Generational.num_gens
.. autoproperty:: wrapper.Generational.checkpoint_enable
.. autoproperty:: wrapper.Generational.checkpoint_freq
.. autoproperty:: wrapper.Generational.checkpoint_filename
.. autoproperty:: wrapper.Generational.verbose
.. autoproperty:: wrapper.Generational.random_seed
.. autoproperty:: wrapper.Generational.logbook
.. autoproperty:: wrapper.Generational.num_evals
.. autoproperty:: wrapper.Generational.runtime
.. autoproperty:: wrapper.Generational.index
.. autoproperty:: wrapper.Generational.container
.. autoproperty:: wrapper.Generational.representatives

Private Properties
------------------
.. autoproperty:: wrapper.Generational._state

Methods
-------
.. automethod:: wrapper.Generational.reset
.. automethod:: wrapper.Generational.evaluate
.. automethod:: wrapper.Generational.best_solutions
.. automethod:: wrapper.Generational.best_representatives
.. automethod:: wrapper.Generational.train
.. automethod:: wrapper.Generational.test

Private methods
---------------
.. automethod:: wrapper.Generational._save_state
.. automethod:: wrapper.Generational._load_state
.. automethod:: wrapper.Generational._new_state
.. automethod:: wrapper.Generational._reset_state
.. automethod:: wrapper.Generational._init_state
.. automethod:: wrapper.Generational._init_internals
.. automethod:: wrapper.Generational._reset_internals
.. automethod:: wrapper.Generational._init_search
.. automethod:: wrapper.Generational._start_generation
.. automethod:: wrapper.Generational._preprocess_generation
.. automethod:: wrapper.Generational._do_generation
.. automethod:: wrapper.Generational._postprocess_generation
.. automethod:: wrapper.Generational._finish_generation
.. automethod:: wrapper.Generational._search
.. automethod:: wrapper.Generational._finish_search
.. automethod:: wrapper.Generational._init_representatives
