IMAGE=torchunet
run:
	docker build -t ${IMAGE} .
in:
	 NV_GPU=2 nvidia-docker run --runtime=nvidia -v /home/hagi:/home/app --rm -it ${IMAGE}:latest bash
