FROM ubuntu:22.04

LABEL MAINTAINER "pne with Python Node.js and Emacs"

ENV USER_NAME=pne
ENV USER_PASSWORD=pnepne
ENV USER_UID=1000
ENV USER_GID=1000
RUN groupadd -g $USER_GID $USER_NAME && \
    useradd -m -s /usr/bin/bash -u $USER_UID -g $USER_GID $USER_NAME && \
    echo "$USER_NAME:$USER_PASSWORD" | chpasswd

RUN set -eux; \
    apt-get update; \
    apt-get upgrade -y; \
    apt-get autoremove

USER $USER_NAME
WORKDIR /home/$USER_NAME
CMD ["bash"]
