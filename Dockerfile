FROM --platform=linux/amd64 curoky/vcpkg:ubuntu21.10
RUN apt update && apt -y install cmake && apt -y install time
RUN vcpkg install benchmark && vcpkg install eigen3 && vcpkg install fmt && vcpkg install nlohmann-json
RUN git clone https://github.com/EDM-Developers/EDM.git /edm

WORKDIR /edm
RUN git checkout arrayfire
RUN export VCPKG_ROOT=/opt/vcpkg && cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
RUN make

WORKDIR /edm/test
# Now we could run ./gbench to start Google Benchmarks test suite
#RUN ./gbench
CMD ["bash"]