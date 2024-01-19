import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import numpy as np #NUMPY!
import math
import cv2
from io import BytesIO


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#if face_cascade.empty(): raise Exception("¿Está seguro que es la ruta correcta?")

#@st.experimental_singleton
#def load_model():
 #   nlp = spacy.load("en_core_med7_lg")
  #  return nlp


def pdf_to_images(pdf_file, resolution=600):
    doc = fitz.open(pdf_file)
    images = []
    for page_number in range(doc.page_count):
        page = doc.load_page(page_number)
        
        # Ajustar la resolución a 300 DPI (puedes cambiar esto según tus necesidades)
        image = page.get_pixmap(matrix=fitz.Matrix(resolution/72, resolution/72))
        
        images.append(Image.frombytes("RGB", [image.width, image.height], image.samples))
    return images

def obtenerDPI(img):
    # Si `img` no es un arreglo NumPy, intenta convertirlo
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    alto, ancho = img.shape[0:2]
    ###dpi hoja
    if int((ancho/8.5)/100)*100 == int((alto/11)/100)*100:
        dpi = int((ancho/8.5)/100)*100
    elif int((ancho/8.5)/100)*100 < int((alto/11)/100)*100:
        dpi = int((ancho/8.5)/100)*100
    else:
        dpi = int((alto/11)/100)*100
    return dpi


def median(l):
    half = len(l) // 2
    l.sort()
    if not len(l) % 2:
        return (l[half - 1] + l[half]) / 2.0
    return l[half]


def centro (xi, yi, xf, yf):
    return (xi + (xf - xi) // 2), (yi + (yf - yi) // 2)


def buscaEsquinas(img, tolerancia, conMarco = False):
    if conMarco:
        marco = tolerancia // 2
    else:
        marco = 0    
    # Si `img` no es un arreglo NumPy, intenta convertirlo
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    alto, ancho = img.shape[0:2]
    img = img[marco:alto-marco,marco:ancho-marco]
    alto, ancho = img.shape[0:2]
    coordenadas = np.where(img  < valorNegro)
    coordenadasX = coordenadas[1]
    coordenadasY = coordenadas[0]
    coordenadasX.sort()
    coordenadasY.sort()
    yS = max(0, coordenadasY[0] - tolerancia)
    yI = min(alto, coordenadasY[-1] + tolerancia)
    xI = max(0,coordenadasX[0] - tolerancia)
    xD = min(ancho, coordenadasX[-1] + tolerancia)
    return xI, xD, yS, yI

def buscaFormato(img, imgOriginal):
    ###tamaño hoja
    # Si `img` no es un arreglo NumPy, intenta convertirlo
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    
    altoImg, anchoImg = img.shape[0:2]
    xI, xD, yS, yI  = buscaEsquinas(img, 10, True)
    
    roi = img[yS:yI, xI:xD]
    roiOriginal = imgOriginal[yS:yI, xI:xD]
    ###rotaci\'on
    ca = abs(yI - yS)
    co = abs(xD - xI)
    angulo = math.atan(ca/co)
    if yS < yI: #fin Y
        angulo = 1 * angulo
    else:
        angulo = -1 * angulo        
    #imagenRotada = imutils.rotate(roi, angulo)
    #centro = (roi.shape[0] // 2, roi.shape[1] // 2)
    aux = Image.fromarray(roi)
    imagenRotada = np.array(aux.rotate(angulo))
    auxOriginal = Image.fromarray(roiOriginal)
    imagenOriginalRotada = np.array(auxOriginal.rotate(angulo))
    return imagenRotada, imagenOriginalRotada


def buscaRegionFotoFirma(img, imgOriginal):
    ###encontrar regi\'on de foto y firma
    # Si `img` no es un arreglo NumPy, intenta convertirlo
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    alto, ancho = img.shape[0:2]
    imagenFotoFirma = img[(alto // 4) * 3:, :]
    imagenOriginalFotoFirma = imgOriginal[(alto // 4) * 3:, :]
    return imagenFotoFirma, imagenOriginalFotoFirma


def buscaFoto(img, imgOriginal):
    ###tamaño foto en pixeles
    wFoto = int(1 * dpi)
    hFoto = int(1.2 * dpi)
    buscaX = wFoto // 8 #division de ancho
    buscaY = hFoto // 10 #division de alto
    ###detectar foto
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            centroX = x + int(w/2)
            centroY = y + int(h/2)
        ###recortar foto
        roi = img[centroY - (buscaY * 5):centroY + (buscaY * 7), centroX - (buscaX * 5):centroX + (buscaX * 5)]
        roiOriginal = imgOriginal[centroY - (buscaY * 5):centroY + (buscaY * 7), centroX - (buscaX * 5):centroX + (buscaX * 5)]
        roiY, roiX = roi.shape[0:2]
        crX = roiX // 2 #centro de la roi
        crY = roiY // 2
        brX = roiX // 16 #division ancho roi
        brY = roiY // 20
        if not isinstance(roi, np.ndarray):
            roi = np.array(roi)
        ret, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        ##rotacion de la roi y busqueda fin en Y 
        cambioI =[]
        cambioD =[]
        for y in range(crY, crY + (brY * 9)):
            aI = int(binary[y - 2, crX - (brX * 5)])
            bI = int(binary[y - 1, crX - (brX * 5)])
            cI = int(binary[y - 0, crX - (brX * 5)])
            dI = int(binary[y + 1, crX - (brX * 5)])
            if (cI + dI) - (aI + bI) > 0:
                cambioI.append(y)
            
            aD = int(binary[y - 2, crX + (brX * 5)])
            bD = int(binary[y - 1, crX + (brX * 5)])
            cD = int(binary[y - 0, crX + (brX * 5)])
            dD = int(binary[y + 1, crX + (brX * 5)])
            if (cD + dD) - (aD + bD) > 0:
                cambioD.append(y)
                
        finI = int(median(cambioI))
        finD = int(median(cambioD))
    
        ca = brX * 10
        co = abs(finD - finI)
        angulo = math.atan(ca/co)
        if finI < finD: #fin Y
            angulo = 1 * angulo
            fin = finI
        else:
            angulo = -1 * angulo
            fin = finD
            
        diferencia = abs(finD - finI)
        # imagenRotada = imutils.rotate(roi, angulo)
        # imagenOriginalRotada = imutils.rotate(roiOriginal, angulo)
        imagenRotada = np.array(Image.fromarray(roi).rotate(angulo))
        imagenOriginalRotada = np.array(Image.fromarray(roiOriginal).rotate(angulo))
        if angulo > 0:
            diferencia = diferencia + 3
        else:
            diferencia = diferencia - 3 
        imagenCortada = imagenRotada[fin - hFoto + diferencia:fin - diferencia, crX - (wFoto // 2) + diferencia:crX + (wFoto // 2) - diferencia]
        imagenOriginalCortada = imagenOriginalRotada[fin - hFoto + diferencia:fin - diferencia, crX - (wFoto // 2) + diferencia:crX + (wFoto // 2) - diferencia]
    else:
        print("no hay foto")
        imagenCortada = img
        imagenOriginalCortada = imgOriginal
        
    return imagenCortada, imagenOriginalCortada


def buscaFirma(img, imgOriginal):
    alto, ancho = img.shape[0:2]
    parteAncho = ancho // 20
    sobranteAncho = (ancho - (parteAncho * 20)) // 2
    parteAlto = alto // 20
    sobranteAlto = (alto - (parteAlto * 20)) // 2 
    auxFirma = img[4*parteAlto-sobranteAlto:alto-7*parteAlto+sobranteAlto, 9*parteAncho-sobranteAncho:ancho-1*parteAncho+sobranteAncho]
    auxOriginalFirma = imgOriginal[4*parteAlto-sobranteAlto:alto-7*parteAlto+sobranteAlto, 9*parteAncho-sobranteAncho:ancho-1*parteAncho+sobranteAncho]
    return auxFirma, auxOriginalFirma
    
def firmaA(img, imgOriginal):
    _, threshold = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    xI, xD, yS, yI  = buscaEsquinas(threshold, -10, False)
    firma = img[yS:yI, xI:xD]
    originalFirma = imgOriginal[yS:yI, xI:xD]
    return firma, originalFirma

def firmaB(img, imgOriginal):
    _, threshold = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    xI, xD, yS, yI  = buscaEsquinas(threshold, 10, False)
    firma = img[yS:yI, xI:xD]
    firmaOriginal = imgOriginal[yS:yI, xI:xD]
    return firma, firmaOriginal
    
def adjust_brightness_contrast(image, brightness, contrast):
    # Convertir la imagen a formato float32
    image = image.astype(np.float32)

    # Aplicar el ajuste de brillo y contraste
    image = image * (contrast/127 + 1) - contrast + brightness

    # Asegurarse de que los valores estén en el rango [0, 255]
    image = np.clip(image, 0, 255)

    # Convertir la imagen de vuelta a tipo uint8
    image = image.astype(np.uint8)

    return image

def eliminar_ruido(img):
    imagen_suavizada = cv2.bilateralFilter(img, 9, 75, 75)  # Ajusta los parámetros según sea necesario
    return imagen_suavizada

def eliminar_ruido_mediana(img):
    imagen_suavizada = cv2.medianBlur(img, 5)  # Ajusta el tamaño de la ventana según sea necesario
    return imagen_suavizada


# Configuración de la página
st.title("Segmentación de foto y firma")
file = st.file_uploader("Selecciona un archivo PDF", type="pdf")
print(file.name)

#nlp = load_model()
valorNegro = 20

# Verificar si se ha subido un archivo
if file is not None:
    # Convertir PDF a imágenes con resolución de 300 DPI
    images = pdf_to_images(file, resolution=300)

    # Mostrar imágenes
    # for i, image in enumerate(images):
    #     print(type(image))
    #     st.image(image, caption=f"Página {i + 1}", use_column_width=True)
    
    ###Proceso
    img = images[0]
    img = np.array(img)  # Convertir a NumPy
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dpi = obtenerDPI(img)
    img2 = np.array(img)  # Convertir a NumPy
    imgLimpia = eliminar_ruido(img2)
    
    imagenRotada, imagenOriginalRotada = buscaFormato(imgLimpia, img2)
    imagenFotoFirma, imagenOriginalFotoFirma = buscaRegionFotoFirma(imagenRotada, imagenOriginalRotada)
    imagenFoto, imagenOriginalFoto = buscaFoto(imagenFotoFirma, imagenOriginalFotoFirma)
    imagenFirma, imagenOriginalFirma = buscaFirma(imagenFotoFirma, imagenOriginalFotoFirma)
    firma, originalFirma = firmaA(imagenFirma, imagenOriginalFirma)
    firmaB, firmaOriginalB = firmaB(firma, originalFirma)
    # Ajustar el brillo y contraste
    brightness = 50
    contrast = 2.0
    adjustedImage = adjust_brightness_contrast(firmaOriginalB, brightness, contrast)
    
    # Dividir la interfaz en tres columnas
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(imagenOriginalFoto, caption="Foto", width=200, use_column_width=False)
        # Convertir la imagen en bytes
        imagen = Image.fromarray(imagenOriginalFoto.astype('uint8'))
        # Convertir la imagen en bytes
        buffer = BytesIO()
        imagen.save(buffer, format="PNG")  # Aquí se usa PNG, pero puedes ajustar el formato según tus necesidades
        # Obtener los bytes de la imagen
        imagen_bytes = buffer.getvalue()
        # Crear un botón de descarga
        st.download_button(
            label="Descargar foto",
            data=imagen_bytes,
            file_name="foto.png",
            key="descargar_btn_foto"
        )
    with col2:
        st.image(firmaOriginalB, caption="Firma Original", width=200, use_column_width=False)
        # Convertir la imagen en bytes
        imagen = Image.fromarray(imagenOriginalFoto.astype('uint8'))
        # Convertir la imagen en bytes
        buffer = BytesIO()
        imagen.save(buffer, format="PNG")  # Aquí se usa PNG, pero puedes ajustar el formato según tus necesidades
        # Obtener los bytes de la imagen
        imagen_bytes = buffer.getvalue()
        # Crear un botón de descarga
        st.download_button(
            label="Descargar firma",
            data=imagen_bytes,
            file_name="firma01.png",
            key="descargar_btn_firma01"
        )
    with col3:
        st.image(adjustedImage, caption="Firma Limpia", width=200, use_column_width=False)
        # Convertir la imagen en bytes
        imagen = Image.fromarray(imagenOriginalFoto.astype('uint8'))
        # Convertir la imagen en bytes
        buffer = BytesIO()
        imagen.save(buffer, format="PNG")  # Aquí se usa PNG, pero puedes ajustar el formato según tus necesidades
        # Obtener los bytes de la imagen
        imagen_bytes = buffer.getvalue()
        # Crear un botón de descarga
        st.download_button(
            label="Descargar firma",
            data=imagen_bytes,
            file_name="firma02.png",
            key="descargar_btn_firma02"
        )
    
