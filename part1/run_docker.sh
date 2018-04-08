#!/bin/bash

docker build -t bu2/ignition .

docker run -it --name ignition --rm -v $PWD/workspace:/home/jovyan/workspace -v '/media/ ... /data':/home/jovyan/data -v $HOME/lab/tsdb/ignition/src:/home/jovyan/src -p 8888:8888 bu2/ignition /bin/bash
