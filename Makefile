install:
	@echo "Installing..."
	poetry install
	poetry run pre-commit install

activate:
	@echo "Activating virtual environment"
	poetry shell

initialize_git:
	@echo "Initialize git"
	git init

setup: initialize_git install

doc:
	@echo "Creating API document"
	rm -f html/test.html
	pdoc --html src/llm/training.py