.PHONY: install run test lint format clean

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

run:
	python main.py

test:
	pytest tests/ -v

lint:
	flake8 . --exclude=venv,__pycache__
	mypy .

format:
	black .
	isort .

clean:
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete

