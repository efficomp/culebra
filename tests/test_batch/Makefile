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
# This work was supported by project PGC2018-098813-B-C31 (Spanish "Ministerio
# de Ciencia, Innovación y Universidades"), and by the European Regional
# Development Fund (ERDF).

# A bash shell will be used for the recipes
SHELL = /bin/bash

# Configuration file
CONFIG = config.toml

# Run script
RUN = run.py

# Checkpoint file
CHECKPOINT = checkpoint.gz

# Backup file
BACKUP = backup.gz

# Results file
RESULTS = results.xlsx

# Experiments
EXPERIMENTS = $(subst /,,$(sort $(wildcard */) ))

# Tools
PYTHON = /usr/bin/env python3
RM = rm -rf

.PHONY: all
all : $(RESULTS)

$(RESULTS): $(RUN)
	@echo "Running ..."
	@./$(RUN)
	@echo

$(RUN): $(CONFIG)
	@echo "Generating the run script ..."
	@echo -e \
"import culebra\\nculebra.tools.batch.Batch.generate_script('$(CONFIG)')" | $(PYTHON)
	@echo

.PHONY: clean
clean:
	@echo "Cleaning ..."
	@$(RM) $(BACKUP) $(RESULTS) $(RUN)
	@echo

.PHONY: distclean
distclean: clean
	@$(RM) $(CHECKPOINT) $(EXPERIMENTS)
