# SIA-TP2

## Librerias Necesarias

Para poder ejecutar el motor es necesario poseer las siguientes librerias de python:

- matplotlib
- numpy

En caso de no tenerlas instaladas, se pueden ejecutar los siguientes comandos para hacerlo:

`pip install matplotlib`

`pip install numpy`

## Ejecución del proyecto

### Ejercicio 1

#### Ejecutar programa

Debe estar en carpeta EJ1/src:

```bash
    python main.py
```

#### Administrar parámetros

Para administrar parámetros hay un archivo config.json,donde podemos gestionar tres parámetros diferentes de nuestro programa.

```json
{
    "operation": "and",
    "learning_rate": 0.001,
    "num_epochs": 100
}
```

Donde la operación puede ser "and" o "xor".
Los otros dos parámetros learning rate y numero de épocas pueden tener cualquier número que consideremos oportuno.

### Ejercicio 2

Para ejecutar el ejercicio 2, uno debe moverse al directorio llamado "EJ2" por medio del siguiente comando mientras se esta ubicado en la carpeta raíz del proyecto:

`cd EJ2`

Luego se puede correr el siguiente comando ubicado dentro del directorio para correr el ejercicio:

`python main.py`
