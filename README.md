# TFG-Contendores-Quemados
Resnet50 para detectar contenedores quemados

Para desplegar con doker descrga el archivo quemados.mar y guradalo en la carpeta modeloDir. Abre la consola de comandos en ese directorio y ejecuta el siguiente comando:
```
docker run --rm -it \
-v $(pwd)/modeloDir:/home/model-server/modeloDir pytorch/torchserve:0.1-cpu \
torchserve --start --modeloDir modeloDir --models quemados=quemados.mar
```
