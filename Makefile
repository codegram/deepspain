deps: requirements.txt
	pip install -r requirements.txt

lint:
	flake8

.PHONY: lint