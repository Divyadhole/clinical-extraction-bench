MODELS ?= ollama:qwen2.5:3b ollama:llama3.2:3b
N      ?= 60

.PHONY: setup fetch bench score report audit all test clean

setup:                ## install deps and pull the models
	pip install -r requirements.txt
	ollama pull qwen2.5:3b
	ollama pull llama3.2:3b

fetch:                ## build the corpus from ClinicalTrials.gov
	python -m ceb.fetch -n $(N)

bench:                ## run every model x prompt (resumable)
	python -m ceb.bench --models $(MODELS)

score:
	python -m ceb.score

report:
	python -m ceb.report

audit:                ## measure how good the free ground truth is
	python -m ceb.audit

all: fetch bench score report

test:
	pytest

clean:
	rm -f results/raw.jsonl results/scores.json
