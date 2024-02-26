ENV_NAME := .venv
PYTHON := python3
REQUIREMENTS := requirements.txt

all: venv

venv: $(ENV_NAME)/bin/activate

$(ENV_NAME)/bin/activate: $(REQUIREMENTS)
	@test -d $(ENV_NAME) || $(PYTHON) -m venv $(ENV_NAME)
	@$(ENV_NAME)/bin/pip install -U pip
	@$(ENV_NAME)/bin/pip install -r $(REQUIREMENTS)
	@touch $(ENV_NAME)/bin/activate

req: $(ENV_NAME)/bin/activate
	@$(ENV_NAME)/bin/pip install -r $(REQUIREMENTS)

clean:
	@rm -rf $(ENV_NAME)
	@echo "Virtual environment removed."

shell:
	@. $(ENV_NAME)/bin/activate; $(SHELL)

.PHONY: all venv requirements clean shell
