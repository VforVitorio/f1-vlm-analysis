IMAGE := vlm-captioning

build:
	docker build -t $(IMAGE) .

run-blip:
	docker run --rm --gpus all \
		-v "$(PWD):/opt/project" \
		-w /opt/project \
		$(IMAGE) python src/vlm_inference.py --model blip

run-moondream:
	docker run --rm --gpus all \
		-v "$(PWD):/opt/project" \
		-w /opt/project \
		$(IMAGE) python src/vlm_inference.py --model moondream

run-minicpm:
	docker run --rm --gpus all \
		-v "$(PWD):/opt/project" \
		-w /opt/project \
		$(IMAGE) python src/vlm_inference.py --model minicpm

run-all:
	docker run --rm --gpus all \
		-v "$(PWD):/opt/project" \
		-w /opt/project \
		$(IMAGE) python src/vlm_inference.py --all

evaluate:
	docker run --rm --gpus all \
		-v "$(PWD):/opt/project" \
		-w /opt/project \
		$(IMAGE) python src/evaluation.py

compare:
	docker run --rm --gpus all \
		-v "$(PWD):/opt/project" \
		-w /opt/project \
		$(IMAGE) python src/comparison.py

shell:
	docker run -it --rm --gpus all \
		-v "$(PWD):/opt/project" \
		-w /opt/project \
		$(IMAGE) /bin/bash

clean:
	docker rmi $(IMAGE)