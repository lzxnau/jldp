FROM ubuntu:22.04

LABEL MAINTAINER "pne with Python Node.js and Emacs"

ENV USER_NAME=pne
ENV USER_PASSWORD=pnepne
ENV USER_UID=1000
ENV USER_GID=1000
ENV TZ=Australia/Sydney

ENV LANG C.UTF-8
ENV GPG_KEY A035C8C19219BA821ECEA86B64E628F8D684696D
ENV PYTHON_VERSION 3.11.6

RUN groupadd -g $USER_GID $USER_NAME && \
    useradd -m -s /usr/bin/bash -u $USER_UID -g $USER_GID $USER_NAME && \
    echo "$USER_NAME:$USER_PASSWORD" | chpasswd

RUN set -eux; \
    apt-get update; \
    apt-get install -y sudo curl git \
        libbluetooth-dev \
        tk-dev \
        uuid-dev \
    ; \
    apt-get upgrade -y; \
    apt-get autoremove; \
    rm -rf /var/lib/apt/lists/*

RUN set -eux; \
        wget -O python.tar.xz \
"https://www.python.org/ftp/python/${PYTHON_VERSION%%[a-z]*}/Python-$PYTHON_VERSION.tar.xz"; \
        wget -O python.tar.xz.asc \
"https://www.python.org/ftp/python/${PYTHON_VERSION%%[a-z]*}/Python-$PYTHON_VERSION.tar.xz.asc"; \
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

RUN usermod -aG sudo $USER_NAME
USER $USER_NAME
WORKDIR /home/$USER_NAME

RUN bash -c "$(curl -fsSL \
https://raw.githubusercontent.com/ohmybash/oh-my-bash/master/tools/install.sh)"

RUN unset USER_NAME USER_PASSWORD USRE_UID USER_GID
CMD ["bash"]
