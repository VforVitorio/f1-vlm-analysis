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

run-swin-tiny:
	docker run --rm --gpus all \
		-v "$(PWD):/opt/project" \
		-w /opt/project \
		$(IMAGE) python3 src/vlm_inference.py --model swin-tiny

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

evaluate-swin-tiny:
	docker run --rm --gpus all \
		-v "$(PWD):/opt/project" \
		-w /opt/project \
		$(IMAGE) python3 src/evaluate_captions.py --model swin-tiny

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