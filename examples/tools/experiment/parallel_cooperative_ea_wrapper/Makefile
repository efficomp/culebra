# This file is part of culebra.
#
# Culebra is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# Culebra is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# Culebra. If not, see <http://www.gnu.org/licenses/>.
#
# This work is supported by projects PGC2018-098813-B-C31 and
# PID2022-137461NB-C31, both funded by the Spanish "Ministerio de Ciencia,
# Innovación y Universidades" and by the European Regional Development Fund
# (ERDF).

# A bash shell will be used for the recipes
SHELL = /bin/bash

# Configuration file
CONFIG_SCRIPT = config.py

# Run script
RUN_SCRIPT = run.py

# Results
RESULTS_BASENAME = results
RESULTS_PICKLED = $(RESULTS_BASENAME).gz
RESULTS_EXCEL = $(RESULTS_BASENAME).xlsx
RESULTS = $(RESULTS_PICKLED) $(RESULTS_EXCEL)

# Temporary files
TEMP = $(RUN_SCRIPT) __pycache__

# Checkpoint files
CHECKPOINTS = $(wildcard checkpoint*)

# Tools
PYTHON = /usr/bin/env python3
RM = rm -rf


.PHONY: all
all : $(RESULTS_EXCEL)

$(RESULTS_EXCEL): $(RUN_SCRIPT)
	@echo "Running ..."
	@$(PYTHON) $(RUN_SCRIPT)
	@echo

$(RUN_SCRIPT): $(CONFIG_SCRIPT)
	@echo "Generating the run script ..."
	@echo -e \
"import culebra\\nculebra.tools.Experiment.generate_run_script('$(CONFIG_SCRIPT)')" | $(PYTHON)
	@echo

.PHONY: clean
clean:
	@echo "Cleaning ..."
	@$(RM) $(RESULTS) $(TEMP)
	@echo

.PHONY: deepclean
deepclean: clean
	@$(RM) $(CHECKPOINTS)
