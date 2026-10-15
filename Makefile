.PHONY: install seed api ui mcp test eval docker

install:
	pip install -r requirements.txt

seed:
	python scripts/seed_store.py

api:
	uvicorn helpdesk.api:app --reload

ui:
	streamlit run app.py

mcp:
	python -m helpdesk.mcp_server

test:
	pytest -q

eval:
	python eval/build_tickets.py
	python eval/run_eval.py --sleep 1

docker:
	docker compose up --build
