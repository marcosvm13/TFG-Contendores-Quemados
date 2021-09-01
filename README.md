# TFG-Evaluación de Contendores De Basura
## Introducción
Se utiliza una arquitectura Resnet50 para resolver un problema multietiqueta de  detección del estado de contenedores. En concreto, el problema se basa en detectar si un contenedor es soterrado, si tien grafitis o si tiene quemaduras. El Experimento 1 se utiliza la pérdida BCE, en el Experimento 2 la pérdida MarginLoss, en el Experimento 3 se añade data augmentation al modelo, en el experimento 4 solo se realiza el experimento sobre el problema de detección de quemados. 

## Despliegue
Para desplegar con doker el modelo de detección de contenedores quemados entrenado con el Experimento 4, descrga el archivo quemados.mar de la siguiente dirección: https://urjc-my.sharepoint.com/:u:/g/personal/m_vazquezm_2017_alumnos_urjc_es/EaiUSp8485pGlO0iK1B-ZSYBtMmU-Ejo8-GsGYLcBYz_lw?e=mhlxwo. Guradalo en la carpeta modeloDir. Abre la consola de comandos en ese directorio y ejecuta el siguiente comando:
```
docker run --rm -it \
-v $(pwd)/modeloDir:/home/model-server/modeloDir pytorch/torchserve:0.1-cpu \
torchserve --start --modeloDir modeloDir --models quemados=quemados.mar
```
