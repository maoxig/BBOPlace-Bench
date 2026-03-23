docker run -it \
  --name mo-bbo \
  --gpus all \
  --rm \
  --shm-size=16g \
  -u $(id -u):$(id -g) \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -v $(realpath ~/project/BBOPlace-Bench):$(realpath ~/project/BBOPlace-Bench) \
  -w $(realpath ~/project/BBOPlace-Bench) \
  --privileged \
  --network host \
  -v $HOME:$HOME \
  -v $HOME/.bashrc:$HOME/.bashrc \
  -v $HOME/.profile:$HOME/.profile \
  crt/bboplace-bench:2.1.0 \
  /bin/bash