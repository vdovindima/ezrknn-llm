FROM ubuntu:22.04 AS builder

RUN apt-get update && apt-get install -y cmake build-essential

WORKDIR /app

COPY . .

WORKDIR /app/examples/DeepSeek-R1-Distill-Qwen-1.5B_Demo/deploy

RUN set -ex; ./build-linux.sh

FROM ubuntu:22.04

RUN set -ex; \
    apt-get update && \
    apt-get install -y libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY rkllm-runtime/Linux/librkllm_api/aarch64/* /usr/lib
COPY rkllm-runtime/Linux/librkllm_api/include/* /usr/local/include

COPY --from=builder /app/examples/DeepSeek-R1-Distill-Qwen-1.5B_Demo/deploy/build/build_linux_aarch64_Release/llm_demo /usr/bin/rkllm

RUN echo "* soft nofile 16384" >> /etc/security/limits.conf
RUN echo "* hard nofile 1048576" >> /etc/security/limits.conf
RUN echo "root soft nofile 16384" >> /etc/security/limits.conf
RUN echo "root hard nofile 1048576" >> /etc/security/limits.conf

ENTRYPOINT ["/usr/bin/rkllm"]
