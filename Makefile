# lint with black
lint:
	black --check --diff .

# format with black
lint-fix:
	black .

# run unit tests
test:
	pytest -v tests/unit