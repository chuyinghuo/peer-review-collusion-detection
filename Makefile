.PHONY: install build compile test

install:
	pip install -e .

build:
	pip install build
	python -m build

compile:
	./cpp/compile_count_cliques_c.sh

test:
	pytest tests/ -v
