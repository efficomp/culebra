..
   This file is part of culebra.

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


Welcome to culebra's documentation!
===================================

.. automodule:: culebra


Attributes:
-----------
.. attribute:: DEFAULT_SIMILARITY_THRESHOLD
    :annotation: = 0.0

    Default similarity threshold for fitnesses.

.. attribute:: DEFAULT_MAX_NUM_ITERS
    :annotation: = 100

    Default maximum number of iterations.

.. attribute:: SERIALIZED_FILE_EXTENSION
    :annotation: = '.dill.gz'

    Extension for files containing serialized objects.

.. attribute:: DEFAULT_CHECKPOINT_ENABLE
    :annotation: = True

    Default checkpointing enablement for a :py:class:`~culebra.abc.Trainer`.

.. attribute:: DEFAULT_CHECKPOINT_FREQ
    :annotation: = 10

    Default checkpointing frequency for a :py:class:`~culebra.abc.Trainer`.

.. attribute:: DEFAULT_CHECKPOINT_BASENAME
     :annotation: = 'checkpoint'

     Default basename for checkpointing files.

.. attribute:: DEFAULT_CHECKPOINT_FILENAME
    :annotation: = 'checkpoint.dill.gz'

    Default checkpointing file name for a :py:class:`~culebra.abc.Trainer`.

.. attribute:: DEFAULT_VERBOSITY
    :annotation: = True

    Default verbosity for a :py:class:`~culebra.abc.Trainer`.

.. attribute:: DEFAULT_INDEX
    :annotation: = 0

    Default :py:class:`~culebra.abc.Trainer` index. Only used within a
    distributed approaches.

..
    .. autodata:: DEFAULT_SIMILARITY_THRESHOLD
    .. autodata:: DEFAULT_MAX_NUM_ITERS
    .. autodata:: SERIALIZED_FILE_EXTENSION
    .. autodata:: DEFAULT_CHECKPOINT_ENABLE
    .. autodata:: DEFAULT_CHECKPOINT_FREQ
    .. autodata:: DEFAULT_CHECKPOINT_BASENAME
    .. autodata:: DEFAULT_CHECKPOINT_FILENAME
    .. autodata:: DEFAULT_VERBOSITY
    .. autodata:: DEFAULT_INDEX


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


References
----------

.. [Gonzalez2019] J. González, J. Ortega, M. Damas, P. Martín-Smith,
   John Q. Gan. *A new multi-objective wrapper method for feature
   selection - Accuracy and stability analysis for BCI*.
   **Neurocomputing**, 333:407-418, 2019.
   https://doi.org/10.1016/j.neucom.2019.01.017.

.. [Gonzalez2021] J. González, J. Ortega, J. J. Escobar, M. Damas.
   *A lexicographic cooperative co-evolutionary approach for feature
   selection*. **Neurocomputing**, 463:59-76, 2021.
   https://doi.org/10.1016/j.neucom.2021.08.003.


.. toctree::
    :hidden:

    abc <abc>
    checker <checker>
    solution <solution>
    fitness_function <fitness_function>
    trainer <trainer>
    tools <tools>
