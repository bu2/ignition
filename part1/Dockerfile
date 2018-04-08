FROM jupyter/all-spark-notebook

USER root

RUN apt-get update && apt-get -yq dist-upgrade \
    && apt-get install -yq --no-install-recommends \
    cmake \
    libboost-all-dev \
    libssl-dev \
    libtool \
    bison \
    flex \
    pkg-config \
    automake \
    g++ \
    git \
    libevent-dev \
    make

RUN cp -fv /usr/local/spark/conf/spark-defaults.conf.template /usr/local/spark/conf/spark-defaults.conf
RUN chown jovyan:100 /usr/local/spark/conf/spark-defaults.conf

USER jovyan
