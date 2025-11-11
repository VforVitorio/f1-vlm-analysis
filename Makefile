IMAGE := f1-vlm-analysis

build:
	docker build -t $(IMAGE) .

run-blip:
	docker run --rm --gpus all \
		-v "$(PWD):/opt/project" \
		-w /opt/project \
		$(IMAGE) python3 src/vlm_inference.py --model blip

run-git-base:
	docker run --rm --gpus all \
		-v "$(PWD):/opt/project" \
		-w /opt/project \
		$(IMAGE) python3 src/vlm_inference.py --model git-base

run-instructblip:
	docker run --rm --gpus all \
		-v "$(PWD):/opt/project" \
		-w /opt/project \
		$(IMAGE) python3 src/vlm_inference.py --model instructblip

run-phi3-vision:
	docker run --rm --gpus all \
		-v "$(PWD):/opt/project" \
		-w /opt/project \
		$(IMAGE) python3 src/vlm_inference.py --model phi3-vision

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

evaluate-git-base:
	docker run --rm --gpus all \
		-v "$(PWD):/opt/project" \
		-w /opt/project \
		$(IMAGE) python3 src/evaluate_captions.py --model git-base

evaluate-instructblip:
	docker run --rm --gpus all \
		-v "$(PWD):/opt/project" \
		-w /opt/project \
		$(IMAGE) python3 src/evaluate_captions.py --model instructblip

evaluate-phi3-vision:
	docker run --rm --gpus all \
		-v "$(PWD):/opt/project" \
		-w /opt/project \
		$(IMAGE) python3 src/evaluate_captions.py --model phi3-vision

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