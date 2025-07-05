import os
import random
import cv2
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread

class Imagenes():
    def __init__(self, ruta_carpeta):
        self.ruta_carpeta = ruta_carpeta
        self.nombres = os.listdir(ruta_carpeta)
        self.estandarizadas = []
        self.stats = {
            'convertidas_a_rgb': 0,
            'ya_en_rgb': 0,
            'total_procesadas': 0
        }

    def mostrar_imagenes_originales(self, cantidad=10):
        """
        Vemos en pantalla las imágenes contenidas en las carpertas de los datasets,
        tomando algunas imágenes aleatorias para demostrar la variedad que hay.
        """

        cantidad = min(cantidad, len(self.nombres))
        seleccionadas = random.sample(self.nombres, cantidad) # Seleccionamos imágenes random a mostrar

        filas = 2
        columnas = (cantidad + 1) // 2

        fig, axes = plt.subplots(filas, columnas, figsize=(15, 6))
        axes = axes.flat if cantidad > 1 else [axes]

        for i, ax in enumerate(axes):
            if i >= cantidad:
                ax.axis('off')
                continue

            ruta_img = os.path.join(self.ruta_carpeta, seleccionadas[i])
            img = cv2.imread(ruta_img, cv2.IMREAD_UNCHANGED)

            if img is None:
                ax.set_title("Imagen no leída")
                ax.axis('off')
                continue

            """
            Ponemos en RGB temporalmente si hay imágenes en BN para la visualización
            """
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f"{seleccionadas[i][:15]}...")

        plt.tight_layout()
        plt.show()

    def obtener_dimensiones_originales(self):
        """
        Calcula y muestra las dimensiones de las imágenes originales en la carpeta, esto para ver cómo
        están primeramente para luego poder redimensionar en caso de que se requiera
        """

        alturas = []
        anchos = []

        for img_name in self.nombres:
            ruta_img = os.path.join(self.ruta_carpeta, img_name)
            try:
                img = imread(ruta_img)
                alturas.append(img.shape[0])
                anchos.append(img.shape[1])
            except Exception as e:
                print(f"No se pudo leer la imagen {img_name}: {e}")

        if alturas and anchos:
            self.alturas = alturas
            self.anchos = anchos
            promedio_alto = np.mean(alturas)
            promedio_ancho = np.mean(anchos)
            print(f"Tamaño promedio: {promedio_alto:.1f} x {promedio_ancho:.1f}")
        else:
            print("No se pudieron calcular dimensiones.")

    def verificar_formatos_color(self):
        """
        Revisa cuántas imágenes están en blanco y negro y cuántas están en color (RGB), esto con el fin de
        poder estandarizar las que no se encuentren en RGB posteriormente
        """

        formatos = {'BN': 0, 'RGB': 0}

        for img_name in self.nombres:
            img_path = os.path.join(self.ruta_carpeta, img_name)
            try:
                # Acá cuenta la cantidad de imágenes que hay en cada categoría (BN / RGB)
                img = imread(img_path)
                if len(img.shape) == 2:
                    formatos['BN'] += 1
                elif len(img.shape) == 3:
                    formatos['RGB'] += 1
            except Exception as e:
                print(f"No se pudo leer la imagen {img_name}: {e}")

        self.formatos = formatos

        # Salida de los resultados
        print(f"Imágenes en blanco y negro: {formatos['BN']}")
        print(f"Imágenes en color (RGB): {formatos['RGB']}")

    def estandarizar_a_rgb(self):
        """
        Lee todas las imágenes de la carpeta y las convierte a RGB si están en B/N,
        guardando las imágenes estandarizadas en una lista nueva para ser utilizada posteriormente
        """

        estandarizadas = [] # Donde se guardan las imágenes ya todas en RGB
        convertidas = 0
        ya_rgb = 0

        for nombre_img in self.nombres:
            ruta_img = os.path.join(self.ruta_carpeta, nombre_img)
            img_original = cv2.imread(ruta_img, cv2.IMREAD_UNCHANGED)

           # En caso de que ocurra algún error al momento de cargar la imagen
            if img_original is None:
                print(f"No se pudo leer la imagen: {nombre_img}")
                continue

            # Acá ya se hace la conversión de BN en caso de que lo requiera a RGB
            if len(img_original.shape) == 2:
                img_rgb = cv2.cvtColor(img_original, cv2.COLOR_GRAY2RGB)
                convertidas += 1
            # Acá solo cuenta las que ya estén con color desde el inicio
            elif len(img_original.shape) == 3 and img_original.shape[2] == 3:
                img_rgb = img_original
                ya_rgb += 1
            else:
                # En caso de que el formato no sea el adecuado, ya que en este caso se está trabajando con .img
                print(f"Formato no reconocido en: {nombre_img}")
                continue

            estandarizadas.append(img_rgb) #Acá ya agrega las imágenes que estén y se vayan convirtiendo a la nueva lista

        self.estandarizadas = estandarizadas
        self.stats = {
            'convertidas_a_rgb': convertidas,
            'ya_en_rgb': ya_rgb,
            'total_estandarizadas': len(estandarizadas)
        }

        # Acá podemos ver un pequeño resumen de la cantidad de imágenes que había ya en RBG, en BN y las que se estandaricen
        print("Imágenes estandarizadas correctamente:")
        print(f"Convertidas de B/N a RGB: {convertidas}")
        print(f"Ya estaban en RGB: {ya_rgb}")
        print(f"Total estandarizadas: {len(estandarizadas)}")

    def redimensionar_imagenes(self, nuevo_tamano=(256, 256)):
        """
        Redimensiona las imágenes ya estandarizadas a RGB al tamaño deseado, usando en este caso el tramaño
        estándar de las imágenes contenidas en las carpetas (256x256)
        """

        # Es mejor primero estandarizar primero, así el color no se afecta tanto ni afecta al momento de redimensionar
        if not hasattr(self, 'estandarizadas') or not self.estandarizadas:
            print("Primero ejecutá .estandarizar_a_rgb() antes de redimensionar.")
            return

        redimensionadas = [] # Nueva lista con las imágenes ya estandarizadas y dimensionadas adecuadamente

        for img in self.estandarizadas:
            img_redim = cv2.resize(img, nuevo_tamano)
            redimensionadas.append(img_redim)

        self.redimensionadas = redimensionadas
        self.tamano_redimensionado = nuevo_tamano

        # Mensaje final del resultado
        print(f"Se redimensionaron {len(redimensionadas)} imágenes a {nuevo_tamano}")