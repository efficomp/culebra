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

   This work was supported by project PGC2018-098813-B-C31 (Spanish "Ministerio
   de Ciencia, Innovación y Universidades"), and by the European Regional
   Development Fund (ERDF).

Configuration file
==================

Experiments can be automated by means of a  `TOML
<https://github.com/toml-lang/toml>`_ configuration file, detailed below,
which is processed by the :py:class:`~tools.config_manager.ConfigManager`
class to build a complete :py:class:`~tools.experiment.Experiment` and even
a :py:class:`~tools.batch.Batch` of experiments.

.. code-block:: cfg
   :linenos:

   # Parameters regarding the batch
   [batch]
     # Number of experiments in the batch. Defaults to 1 if omitted
     n_experiments = <int value>
      
     # Path to create the batch folder. Defaults to the current
     # folder if omitted  
     path = <path>
    
   # Parameters regarding the datset
   [dataset]
     # Parameters regarding the whole dataset
     [dataset.parameters]
       # Index of the output column. Only used if input and ouput
       # columns are in the same file.
       # If input and output columns are in different files this
       # parameter should be omitted
       output_index = <index>

       # Seperator for columns in the files. Defaults to "\\s+" if
       # omitted
       sep = <separator>

       # If this parameter is set, the training dataset will be
       # considered as the only dataset, which will be split into a
       # training and test dataset according to this proportion.
       # The [dataset.test] section will be ignored.
       test_prop = <float value in (0, 1)>

       # If true the input data is normalized in [0, 1]. Defaults to
       # false if omitted
       normalize = <true or false>

       # If this parameter is set, some random features add appended
       # to the dataset
       random_feats = <number of feats to be appended>

       # Seed for the random number generator, defaults to None if
       # omitted
       random_seed = <int value>

     # Parameters regarding the training data
     [dataset.training]

       # Inputs file if input and output columns are seperated in two
       # files. Otherwise this file contains the whole dataset
       file = <path, buffer or URL to the data>

       # Output values. Only used if output_index is omitted
       output_file = <path, buffer or URL to the data>

     # Parameters regarding the test data
     [dataset.test]
       # Inputs file if input and output columns are seperated in two
       # files. Otherwise this file contains the whole dataset
       # Only used if test_prop is omitted
       file = <path, buffer or URL to the data>

       # Output values. Only used if both test_prop and output_index
       # are omitted
       output_file = <path, buffer or URL to the data>

   # Parameters regarding the wrapper method
   [wrapper]
     # Type of wrapper used
     wrapper_cls = <wrapper class>
    
     # Type of individual used
     individual_cls = <individual class>
    
     # Parameters for the individual species
     [wrapper.species]
       # Number of features
       num_feats = <Number of features in the dataset>
    
       # Minimum feature index. Defauts to 0 if omitted
       min_feat = <Minimum feature index>
    
       # Maximum feature index. Defauts to -1 if omitted (the maximum
       # possible feature index)
       max_feat = <Maximum feature index>
    
       # Minimum individual size. Defauts to 0 if omitted
       min_size = <Minimum individual size>
    
       # Maximum individual size. Defauts to -1 if omitted (the
       # maximum possible size)
       max_size = <Maximum individual size>
    
     # Parameters for the wrapper method. These parameters will
     # depend on the wrapper method. Here are some examples.
     [wrapper.parameters]
    
       # Seed for the random number generator, defaults to None if
       # omitted
       random_seed = <int value>
    
       # Whether or not to log the statistics, defaults to true
       verbose = <true or false>
    
       # Checkpoint frequency
       checkpoint_freq = <The checkpoint frequency>
    
       # Checkpoint filepath
       checkpoint_file = <The checkpoint file>
    
       # Population size
       pop_size = <The population size>
    
       # Number of generations
       n_gens = <The number of generations>
    
       # Crossover function. The Individual's default is used if
       # omitted
       xover_func = <Crossover function>
    
       # Crossover probability
       xover_pb = <Crossover probability>
    
       # Mutation function. The Individual's default is used if
       # omitted
       mut_func = <mutation function>
    
       # Mutation probability
       mut_pb = <mutation probability>
    
       # Independent gene mutation probability
       mut_ind_pb = <Independent gene mutation probability>
    
       # Selection function
       sel_func = <selection function>
    
       # Selection function parameters
       [wrapper.parameters.sel_func_params]
         # Parameters for the function

   # Parameters regarding the fitness evaluation
   [fitness]
     # Parameters for the training fitness
     [fitness.training]
       fitness_cls = <Fitness class>

       # Parameters for the fitness evaluation
       [fitness.training.parameters]
         # Configuration parameters for the fitness object (if
         # needed)
         parameter1 = <value1>
         parameter2 = <value2>
          
       # Classifier
       [fitness.training.classifier]
         # Classifier class
         classifier_cls = <Classifier class>

         # Parameters for the classifier
         [fitness.training.classifier.parameters]
           # Configuration parameters for the classifier
           # (if needed)
           parameter1 = <value1>
           parameter2 = <value2>
            
     # Parameters for the test Fitness
     # If omitted, the training fitness will be also used to test
     [fitness.test]
       fitness_cls = <Fitness class>

       # Parameters for the fitness evaluation
       [fitness.test.parameters]
         # Configuration parameters for the fitness object (if
         # needed)
         parameter1 = <value1>
         parameter2 = <value2>

       # Classifier
       [fitness.test.classifier]
         # Classifier class
         classifier_cls = <Classifier class>

         # Parameters for the classifier
         [fitness.test.classifier.parameters]
           # Configuration parameters for the classifier (if needed)
           parameter1 = <value1>
           parameter2 = <value2>
