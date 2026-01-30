.PHONY: dev down logs format lint test

dev:
	docker compose up --build

down:
	docker compose down -v

logs:
	docker compose logs -f --tail=200

format:
	python -m ruff format .

lint:
	python -m ruff check .

test:
	pytest -q
