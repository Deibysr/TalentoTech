# Inteligencia artificial

## Terminología Tecnica
- **Algoritmo:** Secuencia de instrucciones o reglas definidas que son seguidas para realizar una tarea o *resolver un problema*. En IA, los algoritmos pueden aprender de datos.
- **Aprendizaje automático (Machine Learning, ML):** Subcampo de la IA que se centra en el desarrollo de sistemas que pueden aprender de los datos y mejorar su desempeño con el tiempo, sin ser explícitamente programados para cada tarea específica.
- **Redes neuronales:** Modelos computacionales inspirados en el cerebro humano que están diseñados para reconocer patrones. Son un componente clave en muchos sistemas de aprendizaje profundo.
- **Aprendizaje profundo (Deep Learning):** Tipo de aprendizaje automático que utiliza redes neuronales con muchas capas (profundas) para analizar diversos niveles de abstracción de los datos. Es útil en tareas como reconocimiento de voz, visión por computadora y procesamiento del lenguaje natural.
- **Procesamiento de lenguaje natural (PLN):** Rama de la IA que se enfoca en la interacción entre computadoras y humanos a través del lenguaje natural. El objetivo es que las máquinas comprendan y respondan a textos y voces humanas de una manera natural.
- **Visión por computadora:** Campo de la IA que entrena a las computadoras para interpretar y entender el mundo visual. Esto incluye tareas como identificación de objetos, reconocimiento facial y análisis de video.
- **Inteligencia artificial general (AGI):** Un tipo teórico de inteligencia artificial que es capaz de entender, aprender, y aplicar conocimientos en un rango amplio de tareas, de una manera comparable a la inteligencia humana.
- **Sistemas expertos:** Aplicaciones de IA que toman decisiones en áreas específicas, basándose en reglas y datos, imitando el juicio humano experto.
- **Robotica:** Integración de IA con hardware mecánico y otros sistemas para permitir a las máquinas realizar tareas físicas. La robótica utiliza la IA para automatización y mejora de procesos industriales, entre otros.
- **Aprendizaje reforzado:** Método de aprendizaje automático donde un agente aprende a tomar decisiones optimizando acciones basadas en retroalimentación de recompensas y penalizaciones.
- **Datos de entrenamiento:** Conjunto de datos utilizados para entrenar modelos de aprendizaje automático. Estos datos son cruciales para que un modelo aprenda a realizar la tarea deseada correctamente.
- **Sesgo algorítmico:** Sesgos inadvertidos en sistemas de IA que pueden surgir debido a suposiciones en el proceso de modelado, o sesgos en los datos de entrenamiento.

## Impactos de la Inteligencia Artificial
La IA tiene el potencial de transformar industrias enteras, desde la salud hasta la manufactura, pasando por la educación y el entretenimiento. Está cambiando la manera en que interactuamos con la tecnología y cómo esta puede mejorar nuestra eficiencia y capacidad de innovación. Una de las contribuciones más significativas de la IA es su capacidad para automatizar tareas repetitivas o peligrosas, liberando a los humanos para que se enfoquen en actividades más complejas y creativas. Esto no solo aumenta la productividad, sino que también puede mejorar la seguridad en entornos de trabajo peligrosos. Mediante el análisis de grandes volúmenes de datos, la IA puede ayudar a tomar decisiones más informadas y rápidas, especialmente en campos como la medicina, la financiación y la gestión de crisis, donde estos aspectos son cruciales.


## Enlaces
[El futuro de la Inteligencia Artificial](https://www.youtube.com/watch?v=tz8FE5fCrXw&pp=ygUXaW50ZWxpZ2VuY2lhIGFydGlmaWNpYWw%3D)

## Imágenes
![Inteligencia Artificial con Robotica](https://th.bing.com/th/id/OIP.u-uj0RmZlaeosMqrn5KyIQHaE7?rs=1&pid=ImgDetMain)

## Tabla

| Sector         | Impacto de la IA                                                                         |
|----------------|------------------------------------------------------------------------------------------|
| Salud          | - Diagnósticos más rápidos y precisos                                                    |
|                | - Tratamientos personalizados                                                            |
|                | - Automatización de tareas administrativas y repetitivas                                 |
|                | - Desarrollo y descubrimiento acelerado de medicamentos                                  |
|----------------|------------------------------------------------------------------------------------------|
| Manufactura    | - Automatización de líneas de producción                                                  |
|                | - Mantenimiento predictivo de maquinaria                                                 |
|                | - Optimización de cadenas de suministro                                                   |
|                | - Control de calidad mejorado mediante visión artificial                                 |
|----------------|------------------------------------------------------------------------------------------|
| Educación      | - Sistemas de tutoría personalizada                                                       |
|                | - Análisis predictivo para identificar estudiantes en riesgo                              |
|                | - Automatización de tareas administrativas                                                |
|                | - Enriquecimiento de materiales educativos con realidad aumentada y virtual              |
|----------------|------------------------------------------------------------------------------------------|
| Entretenimiento| - Recomendaciones personalizadas de contenido                                            |
|                | - Creación de contenido interactivo y adaptativo                                         |
|                | - Videojuegos con inteligencia artificial que adapta desafíos                             |
|                | - Mejoras en efectos visuales y animación mediante IA                                    |

En este ejemplo, crearemos un modelo de clasificación simple para predecir si una flor es de la especie Iris-setosa basándonos en sus características. Este es un clásico ejemplo introductorio al aprendizaje automático

### Bloque de código con múltiples líneas
```python
# Importamos las bibliotecas necesarias
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Cargamos el conjunto de datos Iris
data = load_iris()
X = data.data  # características
y = data.target  # etiquetas

# Dividimos los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creamos un modelo de árbol de decisión
classifier = DecisionTreeClassifier()

# Entrenamos el modelo
classifier.fit(X_train, y_train)

# Realizamos predicciones en el conjunto de prueba
predictions = classifier.predict(X_test)

# Evaluamos la precisión del modelo
accuracy = accuracy_score(y_test, predictions)
print("Precisión del modelo:", accuracy)
