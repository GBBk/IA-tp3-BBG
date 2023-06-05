import numpy as np
import matplotlib.pyplot as plt

class HopfieldNetwork:
    def __init__(self, num_pixeles):
        self.num_pixeles = num_pixeles
        self.pesos = np.zeros((num_pixeles, num_pixeles))

    def train(self, patrones):
        for patron in patrones:
            patron = np.reshape(patron, (self.num_pixeles, 1))
            self.pesos += np.dot(patron, patron.T)

    def predict(self, patron_entrada, max_iters=100):
        patron_entrada = np.reshape(patron_entrada, (self.num_pixeles, 1))
        patron_salida = np.copy(patron_entrada)

        for _ in range(max_iters):
            patron_salida = np.sign(np.dot(self.pesos, patron_salida))
            if np.array_equal(patron_salida, patron_entrada):
                break

        return patron_salida.flatten()

# Función para agregar ruido a la imagen
def agregar_ruido(imagen, nivel_ruido):
    imagen_ruidosa = np.copy(imagen)
    num_pixeles = imagen.size
    num_pixeles_ruido = int(num_pixeles * nivel_ruido)

    # Generar índices aleatorios para agregar ruido
    indices_ruido = np.random.choice(num_pixeles, num_pixeles_ruido, replace=False)

    # Invertir los píxeles en los índices seleccionados
    imagen_ruidosa.flat[indices_ruido] = 1 - imagen_ruidosa.flat[indices_ruido]

    return imagen_ruidosa

# Imagen original del aro de 10x10 píxeles
imagen_original = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

# Crear una instancia del modelo
red_hopfield = HopfieldNetwork(num_pixeles=100)

# Entrenar el modelo con la imagen original
red_hopfield.train([imagen_original.flatten()])

# Agregar ruido a la imagen original
imagen_ruidosa = agregar_ruido(imagen_original, nivel_ruido=0.1)

# Realizar la denoising
imagen_denoisada = red_hopfield.predict(imagen_ruidosa.flatten())

# Mostrar las imágenes

plt.subplot(131)
plt.title('Imagen Original')
plt.imshow(imagen_original, cmap='gray')

plt.subplot(132)
plt.title('Imagen con Ruido')
plt.imshow(imagen_ruidosa, cmap='gray')

plt.subplot(133)
plt.title('Imagen sin Ruido')
plt.imshow(imagen_denoisada.reshape(10, 10), cmap='gray')

plt.tight_layout()
plt.show()

