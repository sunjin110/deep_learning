deactive:
	deactive

freeze:
	uv pip freeze > requirements.txt

install:
	pip install uv
	uv pip install -r requirements.txt

run:
	@PYTHONPATH=src python3 src/main.py

test:
	PYTHONPATH=src pytest -s ./src/...

mnist_show:
	PYTHONPATH=src python3 src/cmd/mnist_show.py