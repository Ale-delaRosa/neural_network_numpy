# 🌟 Neural Network Numpy 🌟

**👤 Autor:** *Alejandro Garcia*  
**📅 Fecha:** *7 de marzo*  
**📷 Instagram:** [**@ale_garcia454**](https://www.instagram.com/ale_garcia454/)  

🚀 Este proyecto implementa una **red neuronal desde cero** utilizando *Python* y librerías como **NumPy, Matplotlib y Scikit-learn**. La red neuronal se entrena en un conjunto de datos generado artificialmente para **clasificación binaria**.

---

## 🎯 Características
✅ Implementación de una **red neuronal con múltiples capas**.  
✅ Funciones de activación: **Sigmoid y ReLU**.  
✅ Función de pérdida: **Error cuadrático medio (MSE)**.  
✅ Entrenamiento mediante **retropropagación y descenso de gradiente**.  
✅ **Visualización** de los datos de entrenamiento y clasificación de nuevos datos.  

---

## 🔧 Requisitos
Asegúrate de tener instaladas las siguientes librerías antes de ejecutar el código:

```bash
pip install numpy matplotlib scikit-learn
```

---

## 🚀 Uso
Para ejecutar el código, simplemente corre el siguiente comando en tu terminal:

```bash
python main.py
```

Esto **entrenará la red neuronal** y mostrará una gráfica con los datos clasificados.

---

## 📌 Estructura del Código
📌 **create_dataset()**: Genera datos de entrenamiento.  
📌 **sigmoid() y relu()**: Funciones de activación.  
📌 **mse()**: Función de pérdida.  
📌 **initialize_parameters_deep()**: Inicializa los pesos y sesgos de la red.  
📌 **train()**: Realiza la propagación hacia adelante y el entrenamiento de la red.  
📌 **train_neural_network()**: Función principal que **entrena** y **visualiza** los resultados.  

---

## 📜 main.py
Este es el script principal que ejecuta el entrenamiento de la red neuronal:

```python
# Import the neural network training function from the neural_networks_numpy module
from src.neural_network_numpy import train_neural_network

# Check if the script is being executed directly
if __name__ == "__main__":
    train_neural_network()  # Call the function to train the neural network.
```

---

## 📁 Estructura del Proyecto
La estructura del proyecto es la siguiente:

```
📂 main/
│── 📂 src/
│   ├── 📂 __pycache__/
│   ├── 📄 neural_network_numpy.py
│── 📄 .gitignore
│── 📄 README.md
│── 📄 main.py
│── 📄 requirements.txt
```

---

##  Autor
📌 Desarrollado por **Alejandro Garcia**.

