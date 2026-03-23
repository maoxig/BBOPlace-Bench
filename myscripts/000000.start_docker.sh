docker run -it \
  --name mo-bbo \
  --gpus all \
  --rm \
  --network host \
  -u $(id -u):$(id -g) \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -v $(realpath ~/project/BBOPlace-Bench):$(realpath ~/project/BBOPlace-Bench) \
  -w $(realpath ~/project/BBOPlace-Bench) \
  --privileged \
  orfs-dreamplace:latest \
  /bin/bash


