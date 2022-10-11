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

#### Run program

Must be in directory EJ1/src:

```bash
    python main.py
```

#### Manage inputs

To manage inputs there is a config.json file, where we can manage three different parameters of our program.

```json
{
    "operation": "and",
    "learning_rate": 0.001,
    "num_epochs": 100
}
```

Where operation can either be "and" or "xor".
The other two parameters learning rate and number of epochs can have any number that we see fit.

### Ejercicio 2

Para ejecutar el ejercicio 2, uno debe moverse al directorio llamado "EJ2" por medio del siguiente comando mientras se esta ubicado en la carpeta raíz del proyecto:

`cd EJ2`

Luego se puede correr el siguiente comando ubicado dentro del directorio para correr el ejercicio:

`python main.py`
