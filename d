#!/usr/bin/env bash

docker_name="lux_ai"
port_inc=${PORT_INC:=0}

attach() {
  docker exec -w /$docker_name \
    --user $USER \
    -i \
    -t \
    ${USER}_$docker_name \
    /bin/bash
}

start() {
  echo "Ports will be shifted by $port_inc. Use PORT_INC env variable to override it"
  docker run \
    -d \
    -it \
    --gpus all \
    -v ${HOME}/tmp/$docker_name:/tmp/$docker_name \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v `pwd`:/$docker_name \
    -p 0.0.0.0:$((5000 + $port_inc)):$((5000 + $port_inc)) \
    -p 0.0.0.0:$((6006 + $port_inc)):$((6006 + $port_inc)) \
    -p 0.0.0.0:$((8888 + $port_inc)):$((8888 + $port_inc)) \
    --name ${USER}_$docker_name \
    --ipc=host \
    --net=host \
    $docker_name:${USER}_$docker_name
}

build() {
  docker build `dirname $(realpath $0)` \
    -t $docker_name:${USER}_$docker_name \
    --build-arg UID=$(id -u) \
    --build-arg GID=$(id -g) \
    --build-arg USER=$USER \
    --build-arg DISPLAY=$DISPLAY \
    --network=host
}

stop() {
  docker stop ${USER}_$docker_name
  docker rm ${USER}_$docker_name
}

remove_docker() {
  docker rmi ${USER}_$docker_name
}

status() {
  docker ps | grep $docker_name | cat
}

print_help() {
  # using here doc feature
  # https://stackoverflow.com/a/23930212
  cat << END
usage: d [-h] [-u] [-a] [-s] [-b] [-c]

Script to control docker

optional arguments:
  -h, --help            show this help message and exit

commands:
  Various commands for ./d. Could be combined (e.g. ./d -sua)

  -u, --up              Start new docker container
  -a, --attach          Attach (start interactive shell) to running container
  -s, --stop            Stop container if running
  -b, --build           Build new docker images
  -q, --status          Query container status
  -c, --cleanup         Cleanup docker container files. Useful when you change build.
END
}

main() {
  # modeled after our sdc d utility
  # parse command line arguments
  # Combines this tutorials:
  # https://sookocheff.com/post/bash/parsing-bash-script-arguments-with-shopts/
  # https://wiki.bash-hackers.org/howto/getopts_tutorial
  while getopts ":huabscq" opt; do
    case ${opt} in
      h )
        print_help
        exit 0
        ;;
      u )
        echo "Start container"
        start
        ;;
      a )
        echo "Attach to container"
        attach
        ;;
      b )
        echo "Build container"
        build
        ;;
      s )
        echo "Stop container"
        stop
        ;;
      c )
        echo "Remove docker"
        remove_docker
        ;;
      q )
        echo "Query container status"
        status
        ;;
      \? )
        echo "Invalid Option: -${OPTARG}" 1>&2
        exit 1
        ;;
    esac
  done
}

main "$@"
