# Use nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04 or later for CUDA.
# Warning: All ARGs placed before FROM will only be scoped up unitl FROM statement.
# https://github.com/docker/cli/blob/3c7ede6a68941f64c3a154c9a753eb7a9b1c2c3e/docs/reference/builder.md#understand-how-arg-and-from-interact
ARG base_image="debian:buster"
FROM "${base_image}"

ARG python_version="3.7"
ARG release_version="nightly"
ARG cuda="0"
ARG cuda_compute="7.0,7.5"
ARG cxx_abi="0"
ARG tpuvm=""
ARG bazel_jobs=""
ARG git_clone="false"

# RUN rm /etc/apt/sources.list.d/cuda.list
# RUN rm /etc/apt/sources.list.d/nvidia-ml.list
# RUN apt-key del 7fa2af80

# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# COPY ./docker/cuda-keyring_1.0-1_all.deb cuda-keyring_1.0-1_all.deb 

# RUN https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb

# RUN dpkg -i cuda-keyring_1.0-1_all.deb

# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
# RUN dpkg -i cuda-keyring_1.0-1_all.deb

RUN apt-get update

RUN apt-get install -y git sudo python-pip python3-pip
RUN git clone https://github.com/pytorch/pytorch

# Disable CUDA for PyTorch
ENV USE_CUDA "0"

# Enable CUDA for XLA
ENV XLA_CUDA "${cuda}"
ENV TF_CUDA_COMPUTE_CAPABILITIES "${cuda_compute}"

# Whether to build torch and torch_xla libraries with CXX ABI
ENV _GLIBCXX_USE_CXX11_ABI "${cxx_abi}"
ENV CFLAGS "${CFLAGS} -D_GLIBCXX_USE_CXX11_ABI=${cxx_abi}"
ENV CXXFLAGS "${CXXFLAGS} -D_GLIBCXX_USE_CXX11_ABI=${cxx_abi}"

# Whether to build for TPUVM mode
ENV TPUVM_MODE "${tpuvm}"

# Maximum number of jobs to use for bazel build
ENV BAZEL_JOBS "${bazel_jobs}"

# To get around issue of Cloud Build with recursive submodule update
# clone recursively from pytorch/xla if building docker image with
# cloud build. Otherwise, just use local.
# https://github.com/GoogleCloudPlatform/cloud-builders/issues/435


COPY . /pytorch/xla

RUN if [ "${git_clone}" = "true" ]; then github_branch="${release_version}" && \
  if [ "${release_version}" = "nightly" ]; then github_branch="master"; fi && \
  cd /pytorch && \
  rm -rf xla && \
  git clone -b "${github_branch}" --recursive git@github.com:ftxj/xla.git && \
  cd / && \
  git clone -b "${github_branch}" --recursive https://github.com/pytorch-tpu/examples tpu-examples; fi

RUN ls

# COPY cudnn-linux-x86_64-8.4.0.27_cuda10.2-archive.tar.xz /pytorch/xla/cudnn-linux-x86_64-8.4.0.27_cuda10.2-archive.tar.xz

ENV CUDNN_TGZ_PATH "/pytorch/xla/cudnn-linux-x86_64-8.4.0.27_cuda10.2-archive.tar.xz"

RUN cd /pytorch && bash xla/scripts/build_torch_wheels.sh ${python_version} ${release_version}

# Use conda environment on startup or when running scripts.
RUN echo "conda activate pytorch" >> ~/.bashrc
RUN echo "export TF_CPP_LOG_THREAD_ID=1" >> ~/.bashrc
ENV PATH /root/anaconda3/envs/pytorch/bin/:/root/bin:$PATH

# Define entrypoint and cmd
COPY docker/docker-entrypoint.sh /usr/local/bin
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["bash"]
