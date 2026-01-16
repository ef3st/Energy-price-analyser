run: 
	poetry run python src/energy_price_analyser/main.py 

.PHONY: install test unit integration lint format shell

install:
	poetry install

test:
	poetry run pytest -m  "not integration"

test-all:
	poetry run pytest 

test-ns:
	poetry run pytest -m "not slow and not integration"

unit:
	poetry run pytest -m unit

integration:
	poetry run pytest -m integration

lint:
	poetry run flake8 .

format:
	poetry run black .

shell:
	poetry shell