docker run -it \
  --name mo-bbo \
  --gpus all \
  --rm \
  --shm-size=32g \
  -u $(id -u):$(id -g) \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -v $(realpath .):/workspace \
  -w /workspace \
  --privileged \
  --network host \
  -v $HOME:$HOME \
  -v $HOME/.bashrc:$HOME/.bashrc \
  -v $HOME/.profile:$HOME/.profile \
  crt/bboplace-bench:2.1.0 \
  /bin/bash