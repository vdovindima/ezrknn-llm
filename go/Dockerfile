FROM golang:1.25.1-trixie AS builder

RUN apt-get update && apt-get install -y cmake build-essential

WORKDIR /app

COPY main.go main.go
COPY libwrapper.a libwrapper.a
COPY librkllmrt.so librkllm.so
COPY rkllm.h  /usr/local/include

RUN set -ex; go mod init rkllmgo
RUN set -ex; go mod tidy
RUN set -ex; go list -f '{{.CFiles}}'
RUN set -ex; GOOS=linux GOARCH=arm64 CGO_ENABLED=1 go build -x -v -o /app/main main.go

FROM ubuntu:22.04

RUN set -ex; \
    apt-get update && \
    apt-get install -y libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY librkllmrt.so /usr/lib
COPY rkllm.h  /usr/local/include

COPY --from=builder /app/main /usr/bin/rkllm

RUN echo "* soft nofile 16384" >> /etc/security/limits.conf
RUN echo "* hard nofile 1048576" >> /etc/security/limits.conf
RUN echo "root soft nofile 16384" >> /etc/security/limits.conf
RUN echo "root hard nofile 1048576" >> /etc/security/limits.conf

ENTRYPOINT ["/usr/bin/rkllm"]
