'''
Detecção de faces é a capacidade de detectar rostos em uma imagem. Para isso técnicas de visão computacional
buscam em uma imagem características que generalizem de maneira geral a forma de um  rosto. Uma vez que uma face
é detectada em uma imagem o  processo de reconhecimento pode ser iniciado.

########################################TESTES########################################
O algoritmo abaixo realiza testes para Leitura de imagens armazenadas em uma pasta e leitura de uma imagem com detecção de rostos,
extraindo e salvando em uma janela pop-up as faces presentes na imagem.

'''
#########Leitura de Imagens##########
import inline as inline
import matplotlib as matplotlib

matplotlib
inline
import cv2
import matplotlib.pyplot as plt

# Leitura da Imagem

img = cv2.imread('carter.jpg')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

img2 = cv2.imread('DoutorEstranho.jpg')
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.show()

img3 = cv2.imread('Joker2.jpg')
plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
plt.show()

img4 = cv2.imread('MCU.jpg')
plt.imshow(cv2.cvtColor(img4, cv2.COLOR_BGR2RGB))
plt.show()

img5 = cv2.imread('originalJoker.jpg')
plt.imshow(cv2.cvtColor(img5, cv2.COLOR_BGR2RGB))
plt.show()

###############Detecção de Faces##############

def detectar_face(img):
    import cv2
    from matplotlib import pyplot as plt

    # Dimensão das Imagens
    plt.rcParams['figure.figsize'] = (224, 224)

    # Classificador construído para detectar faces
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    img = cv2.imread('MCU.jpg')

    # Transformando a imagem em escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Processo para detectar as faces na imagem
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    cont = 1
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        cont = cont + 1
        cv2.imwrite('resultado/saída.py' + str(cont) + '.jpg', roi_color)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


detectar_face(img)



