IMAGE := f1-vlm-analysis

build:
	docker build -t $(IMAGE) .

run-blip:
	docker run --rm --gpus all \
		-v "$(PWD):/opt/project" \
		-w /opt/project \
		$(IMAGE) python3 src/vlm_inference.py --model blip

run-moondream:
	docker run --rm --gpus all \
		-v "$(PWD):/opt/project" \
		-w /opt/project \
		$(IMAGE) python3 src/vlm_inference.py --model moondream

run-minicpm:
	docker run --rm --gpus all \
		-v "$(PWD):/opt/project" \
		-w /opt/project \
		$(IMAGE) python3 src/vlm_inference.py --model minicpm

run-all:
	docker run --rm --gpus all \
		-v "$(PWD):/opt/project" \
		-w /opt/project \
		$(IMAGE) python3 src/vlm_inference.py --all

evaluate-blip:
	docker run --rm --gpus all \
		-v "$(PWD):/opt/project" \
		-w /opt/project \
		$(IMAGE) python3 src/evaluate_captions.py --model blip

evaluate-moondream:
	docker run --rm --gpus all \
		-v "$(PWD):/opt/project" \
		-w /opt/project \
		$(IMAGE) python3 src/evaluate_captions.py --model moondream

evaluate-minicpm:
	docker run --rm --gpus all \
		-v "$(PWD):/opt/project" \
		-w /opt/project \
		$(IMAGE) python3 src/evaluate_captions.py --model minicpm

evaluate-all:
	docker run --rm --gpus all \
		-v "$(PWD):/opt/project" \
		-w /opt/project \
		$(IMAGE) python3 src/evaluate_captions.py --all

compare:
	docker run --rm --gpus all \
		-v "$(PWD):/opt/project" \
		-w /opt/project \
		$(IMAGE) python3 src/evaluate_captions.py --compare

shell:
	docker run -it --rm --gpus all \
		-v "$(PWD):/opt/project" \
		-w /opt/project \
		$(IMAGE) /bin/bash

clean:
	docker rmi $(IMAGE)