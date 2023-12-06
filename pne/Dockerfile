FROM buildpack-deps:bookworm

LABEL MAINTAINER "PNE with Python Node.js and Emacs"

# Version:
ENV PYTHON_VERSION=3.11.6
ENV NODE_VERSION=20.10.0
ENV EMACS_VERSION=29.1

# Download:
ENV PYLINK=https://www.python.org/ftp/python/
RUN wget -O python.tar.xz \
         "$PYLINK${PYTHON_VERSION%%[a-z]*}/Python-$PYTHON_VERSION.tar.xz"
RUN wget -O python.tar.xz.asc \
         "$PYLINK${PYTHON_VERSION%%[a-z]*}/Python-$PYTHON_VERSION.tar.xz.asc"
RUN unset PYLINK

ENV NJLINK=https://nodejs.org/dist/v
ENV ARCH=x64
RUN curl -fsSLO --compressed \
         "$NJLINK$NODE_VERSION/node-v$NODE_VERSION-linux-$ARCH.tar.xz"
RUN curl -fsSLO --compressed \
         "$NJLINK$NODE_VERSION/SHASUMS256.txt.asc"
RUN unset NJLINK

RUN curl https://ftp.gnu.org/gnu/emacs/emacs-$EMACS_VERSION.tar.gz \
         | tar -xzv

# Pre-installation:
RUN apt update -y
################################################################################

# Install Python:
ENV PATH /usr/local/bin:$PATH
ENV LANG C.UTF-8
ENV GPG_KEY A035C8C19219BA821ECEA86B64E628F8D684696D

RUN apt install -y --no-install-recommends \
        libbluetooth-dev tk-dev uuid-dev

RUN set -eux; \
        GNUPGHOME="$(mktemp -d)"; export GNUPGHOME; \
        gpg --batch --keyserver \
        hkps://keys.openpgp.org --recv-keys "$GPG_KEY"; \
        gpg --batch --verify python.tar.xz.asc python.tar.xz; \
        gpgconf --kill all; \
        rm -rf "$GNUPGHOME" python.tar.xz.asc; \
        mkdir -p /usr/src/python; \
        tar --extract --directory /usr/src/python --strip-components=1 \
            --file python.tar.xz; \
        rm python.tar.xz; \
        cd /usr/src/python; \
        gnuArch="$(dpkg-architecture --query DEB_BUILD_GNU_TYPE)"; \
        ./configure \
                --build="$gnuArch" \
                --enable-loadable-sqlite-extensions \
                --enable-optimizations \
                --enable-option-checking=fatal \
                --enable-shared \
                --with-lto \
                --with-system-expat \
                --without-ensurepip \
        ; \
        nproc="$(nproc)"; \
        EXTRA_CFLAGS="$(dpkg-buildflags --get CFLAGS)"; \
        LDFLAGS="$(dpkg-buildflags --get LDFLAGS)"; \
        make -j "$nproc" \
                "EXTRA_CFLAGS=${EXTRA_CFLAGS:-}" \
                "LDFLAGS=${LDFLAGS:-}" \
                "PROFILE_TASK=${PROFILE_TASK:-}" \
        ; \
        rm python; \
        make -j "$nproc" \
                "EXTRA_CFLAGS=${EXTRA_CFLAGS:-}" \
                "LDFLAGS=${LDFLAGS:--Wl},-rpath='\$\$ORIGIN/../lib'" \
                "PROFILE_TASK=${PROFILE_TASK:-}" \
                python \
        ; \
        make install; \
        bin="$(readlink -ve /usr/local/bin/python3)"; \
        dir="$(dirname "$bin")"; \
        mkdir -p "/usr/share/gdb/auto-load/$dir"; \
        cp -vL Tools/gdb/libpython.py "/usr/share/gdb/auto-load/$bin-gdb.py"; \
        cd /; \
        rm -rf /usr/src/python; \
        find /usr/local -depth \
          \( \
          \( -type d -a \( -name test -o -name tests -o -name idle_test \) \) \
         -o \( -type f -a \( -name '*.pyc' -o -name '*.pyo' -o -name \
         'libpython*.a' \) \) \
         \) -exec rm -rf '{}' + \
        ; \
        ldconfig; \
        python3 --version

RUN set -eux; \
        for src in idle3 pydoc3 python3 python3-config; do \
                dst="$(echo "$src" | tr -d 3)"; \
                [ -s "/usr/local/bin/$src" ]; \
                [ ! -e "/usr/local/bin/$dst" ]; \
                ln -svT "$src" "/usr/local/bin/$dst"; \
        done

ENV PYTHON_PIP_VERSION 23.3.1
ENV PYTHON_SETUPTOOLS_VERSION 69.0.2
ENV PYTHON_WHEEL_VERSION 0.42.0
ENV PIP1=https://github.com/pypa/get-pip/raw/
ENV PIP2=4cfa4081d27285bda1220a62a5ebf5b4bd749cdb/public/get-pip.py
ENV PYTHON_GET_PIP_URL $PIP1$PIP2
ENV PIP3=9cc01665956d22b3bf057ae8287b035827bfd895da235bcea200ab3b811790b6
ENV PYTHON_GET_PIP_SHA256 $PIP3

RUN set -eux; \
        wget -O get-pip.py "$PYTHON_GET_PIP_URL"; \
        echo "$PYTHON_GET_PIP_SHA256 *get-pip.py" | sha256sum -c -; \
        export PYTHONDONTWRITEBYTECODE=1; \
        python get-pip.py \
               --disable-pip-version-check \
               --no-cache-dir \
               --no-compile \
               "pip==$PYTHON_PIP_VERSION" \
               "setuptools==$PYTHON_SETUPTOOLS_VERSION" \
               "wheel==$PYTHON_WHEEL_VERSION" \
        ; \
        rm -f get-pip.py; \
        pip --version

RUN unset PYTHON_VERSION PYTHON_PIP_VERSION PYTHON_SETUPTOOLS_VERSION \
          PYTHON_WHEEL_VERSION PIP1 PIP2 PIP3 PYTHON_GET_PIP_URL \
          PYTHON_GET_PIP_SHA256
################################################################################

# Install Node.js:
RUN export GNUPGHOME="$(mktemp -d)" \
  && set -ex \
  && for key in \
    4ED778F539E3634C779C87C6D7062848A1AB005C \
    141F07595B7B3FFE74309A937405533BE57C7D57 \
    74F12602B6F1C4E913FAA37AD3A89613643B6201 \
    DD792F5973C6DE52C432CBDAC77ABFA00DDBF2B7 \
    61FC681DFB92A079F1685E77973F295594EC4689 \
    8FCCA13FEF1D0C2E91008E09770F7A9A5AE15600 \
    C4F0DFFF4E8C1A8236409D08E73BC641CC11F4C8 \
    890C08DB8579162FEE0DF9DB8BEAB4DFCF555EF4 \
    C82FA3AE1CBEDC6BE46B9360C43CEC45C17AB93C \
    108F52B48DB57BB0CC439B2997B01419BD92F80A \
    A363A499291CBBC940DD62E41F10027AF002F8B0 \
  ; do \
      gpg --batch --keyserver hkps://keys.openpgp.org --recv-keys "$key" || \
      gpg --batch --keyserver keyserver.ubuntu.com --recv-keys "$key" ; \
  done \
  && gpg --batch --decrypt --output SHASUMS256.txt SHASUMS256.txt.asc \
  && gpgconf --kill all \
  && rm -rf "$GNUPGHOME" \
  && grep " node-v$NODE_VERSION-linux-$ARCH.tar.xz\$" SHASUMS256.txt | \
  sha256sum -c - \
  && tar -xJf "node-v$NODE_VERSION-linux-$ARCH.tar.xz" -C /usr/local \
  --strip-components=1 --no-same-owner \
  && rm "node-v$NODE_VERSION-linux-$ARCH.tar.xz" SHASUMS256.txt.asc \
  SHASUMS256.txt \
  && ln -s /usr/local/bin/node /usr/local/bin/nodejs \
  && node --version \
  && npm --version

RUN unset NODE_VERSION ARCH
################################################################################

# Install Emacs:
RUN apt install -y --no-install-recommends \
        libgccjit-12-dev libgnutls28-dev libjansson-dev

RUN cd /emacs-$EMACS_VERSION && ./autogen.sh && \
       ./configure --with-native-compilation --without-sound --without-x \
       --without-compress-install --with-json --with-mailutils

RUN cd /emacs-$EMACS_VERSION && make -j 4 && make install

RUN unset EMACS_VERSION
################################################################################

# Install Packages:
RUN apt install -y --no-install-recommends sudo pandoc

# Create User pne:
ENV USER_NAME=pne
ENV USER_PASSWORD=pnepne
ENV USER_UID=1000
ENV USER_GID=1000

RUN groupadd -g $USER_GID $USER_NAME && \
    useradd -m -s /usr/bin/bash -u $USER_UID -g $USER_GID $USER_NAME && \
    echo "$USER_NAME:$USER_PASSWORD" | chpasswd
RUN usermod -aG sudo $USER_NAME
RUN unset USER_NAME USER_PASSWORD USRE_UID USER_GID

# User pne setup:
USER pne
WORKDIR /home/pne
RUN mkdir .emacs.d env jldp .ssh
RUN bash -c "$(curl -fsSL \
https://raw.githubusercontent.com/ohmybash/oh-my-bash/master/tools/install.sh)"

# Post-installation:
USER root
RUN npm install -g npm@10.2.4 bash-language-server @mermaid-js/mermaid-cli

ARG UDATE=1
RUN apt update && apt upgrade -y && apt autoremove
RUN rm -rf /var/lib/apt/lists/*
RUN npm -g update

USER pne

CMD ["bash"]
