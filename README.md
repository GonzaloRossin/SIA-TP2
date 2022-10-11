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

#### Software Necesario
Para poder observar la evolución del vector W durante el entrenamiento es necesario del siguiente software:
 - ovito: https://www.ovito.org/windows-downloads/

#### Ejecución del proyecto

Para ejecutar el ejercicio 2, uno debe moverse al directorio llamado "EJ2" por medio del siguiente comando mientras se esta ubicado en la carpeta raíz del proyecto:

`cd EJ2`

Para correr el ejercicio se utiliza el siguiente comando ubicado dentro del directorio para correr el ejercicio:

`python main.py`

Es posible cambiar los parámetros de entrenamiento del perceptrón por medio del csv llamado "parameters.csv" ubicado en el directorio raíz del ejercico. Algunos de los parámetros que es posible editar son:

- etha
- beta
- iterations: cantidad de iteraciones/epocas
- training_percentage: porcentage utilizado para el entrenamiento del perceptrón
- activationType: este parámetro modifica la función de activación a utilizar en el perceptrón. Este parámetro posee valores fijos (`linear`, `tanh`, `logistic`)
- trainingType: este parámetro modifica la forma en la que se selecciona el "batch" para entrenar el preceptrón. Este parámetro posee valores fijos (`epoca`, `random`)

El ultimo parámetro "graph_option" determina el gráfico que se quiere generar. El parámetro puede tomar los siguientes valores:
- `1`:  genera el gráfico de error cuadratico medio en función del porcentaje de entrenamiento. Para este gráfico, el parámetro "training_percentage" y "activationType" del csv no serán utilizados. El gráfico generado utilizan las sigmoideas(tanh y logistic) como función de activación
- `2`: genera el gráfico de error cuadratico medio en función de las epocas durante el entrenamiento
- `3`: genera un archivo .xyz con la evolución del vector W en función de las iteraciones durante el entrenamiento. De esta manera es posible ver como el modelo evoluciona para inferir un modelo en base a los parametros utilizados.

#### Instrucciones de uso de ovito
Una vez generado el archivo .xyz, se debe abrir este archivo por medio del software ovito. Una vez abierto el archivo, se abrirá un menú en el cual es posible mapear las columnas del archivo. Las columnas deben estar mapeadas de la siguiente forma:
 - columna 1: position (eje x)
 - columna 2: position (eje y)
 - columna 3: position (eje z)
 - columna 4: charge
 
 Por ultimo, debemos agregar la modificación "color coding" ajustando el rango de color entre 0 y 89 para asi poder ver la evolución por medio del cambio de color de las particulas( las coordenadas representan E1, E2, E3 mientras que el color representa el resultado del modelo evaluado en sus respectivos inputs)
