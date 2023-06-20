#!/bin/bash

echo "Are you in the correct folder? (y/n)"
read ANSWER
case $ANSWER in
    [Yy]* ) echo "Input the name of the project:"
    read NAME
    echo "Welcome to "$NAME"!"
    docker build . -t "$NAME"
    docker run -it -p 8899:8899 -d --mount type=bind,source="$(pwd)",target=/working --name "$NAME" "$NAME"

    docker exec -it "$NAME" jupyter lab --port 8899 --ip=0.0.0.0 --allow-root
    echo "Ok, let's Jupiter em' all!"
esac