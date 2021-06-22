FROM --platform=linux/amd64 curoky/vcpkg:ubuntu21.10

RUN apt update && apt -y install cmake

RUN vcpkg install eigen3 && vcpkg install fmt && vcpkg install nlohmann-json

RUN git clone https://github.com/EDM-Developers/EDM.git /edm

WORKDIR /edm

RUN git checkout for-arrayfire-engineers

RUN export VCPKG_ROOT=/opt/vcpkg && ./compile.bat

WORKDIR /edm/test

RUN chmod +x ./run-tests-for-arrayfire.sh && apt -y install time

# Now we could run ./run-tests-for-arrayfire.sh to get the time taken for each test
#RUN ./run-tests-for-arrayfire.sh

CMD ["bash"]