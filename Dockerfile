FROM --platform=linux/amd64 curoky/vcpkg:ubuntu20.04

# Install the CUDA toolkit

RUN apt update && apt install -y gnupg2 ca-certificates apt-utils software-properties-common wget

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
RUN mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
RUN apt-get update && apt-get -y install cuda

# Install ArrayFire library

RUN apt-key adv --fetch-key https://repo.arrayfire.com/GPG-PUB-KEY-ARRAYFIRE-2020.PUB
RUN echo "deb [arch=amd64] https://repo.arrayfire.com/ubuntu focal main" | tee /etc/apt/sources.list.d/arrayfire.list
RUN apt update && apt install -y arrayfire

# Install the C++ package dependencies using vcpkg

RUN apt update && apt -y install cmake && apt -y install time
RUN vcpkg install benchmark && vcpkg install catch2 && vcpkg install fmt && vcpkg install nlohmann-json

# Compile the EDM plugin

RUN git clone https://github.com/EDM-Developers/EDM.git /edm

WORKDIR /edm

RUN export VCPKG_ROOT=/opt/vcpkg && cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DEDM_WITH_ARRAYFIRE=ON -DArrayFire_DIR=/usr/share/ArrayFire/cmake && cmake --build build --target edm_plugin

CMD ["bash"]