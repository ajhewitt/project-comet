.PHONY: run ci lint test fix
run: ./bin/comet-run
ci: ./bin/ci
lint:
	micromamba run -n comet ruff format --check .
	micromamba run -n comet ruff check .
fix:
	micromamba run -n comet ruff format .
