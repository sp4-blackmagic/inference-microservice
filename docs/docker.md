# Docker commands

The image is being continously updated and build with github actions, you can get it with command:  

```shell
docker pull ghcr.io/sp4-blackmagic/inference_microservice:latest
```

### Build container

```shell
docker build -t inference-service .
```

### Run

```shell
docker run -p 6969:6969 inference-service
```