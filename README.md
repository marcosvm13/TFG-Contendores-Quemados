# TFG-Contendores-Quemados
Resnet50 para detectar contenedores quemados

Para desplegar con doker descrga el archivo quemados.mar de la siguiente dirección: https://urjc-my.sharepoint.com/:u:/g/personal/m_vazquezm_2017_alumnos_urjc_es/EaiUSp8485pGlO0iK1B-ZSYBtMmU-Ejo8-GsGYLcBYz_lw?e=mhlxwo. Guradalo en la carpeta modeloDir. Abre la consola de comandos en ese directorio y ejecuta el siguiente comando:
```
docker run --rm -it \
-v $(pwd)/modeloDir:/home/model-server/modeloDir pytorch/torchserve:0.1-cpu \
torchserve --start --modeloDir modeloDir --models quemados=quemados.mar
```
