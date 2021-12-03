FROM --platform=linux/amd64 nvcr.io/nvidia/cuda:11.4.2-devel-ubuntu20.04

# Install ArrayFire library

RUN apt update && DEBIAN_FRONTEND="noninteractive" apt install -y gnupg2 ca-certificates apt-utils software-properties-common wget

RUN apt-key adv --fetch-key https://repo.arrayfire.com/GPG-PUB-KEY-ARRAYFIRE-2020.PUB
RUN echo "deb [arch=amd64] https://repo.arrayfire.com/ubuntu focal main" | tee /etc/apt/sources.list.d/arrayfire.list
RUN apt update && apt install -y arrayfire

# Install the C++ package dependencies using vcpkg

RUN apt update && apt -y install cmake time git curl zip unzip tar pkg-config

RUN git clone https://github.com/Microsoft/vcpkg.git
RUN ./vcpkg/bootstrap-vcpkg.sh

RUN ./vcpkg/vcpkg install benchmark && ./vcpkg/vcpkg install catch2 && ./vcpkg/vcpkg install fmt && ./vcpkg/vcpkg install nlohmann-json

# Compile the EDM plugin

COPY . /edm
WORKDIR /edm

RUN export VCPKG_ROOT=/vcpkg && cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DEDM_WITH_ARRAYFIRE=ON -DArrayFire_DIR=/usr/share/ArrayFire/cmake && cmake --build build

CMD ["bash"]