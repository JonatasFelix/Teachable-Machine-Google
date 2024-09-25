import numpy as np
from keras.models import load_model
from keras.layers import BatchNormalization, DepthwiseConv2D
from src.image_processing import load_and_preprocess_image

def custom_depthwise_conv2d(**kwargs):
    if 'groups' in kwargs:
        del kwargs['groups']
    return DepthwiseConv2D(**kwargs)

def load_model_and_classes(model_path, labels_path):
    """
    Carrega o modelo Keras e os nomes das classes.

    Args:
        model_path (str): Caminho para o arquivo do modelo.
        labels_path (str): Caminho para o arquivo de labels.

    Returns:
        model: Modelo Keras carregado.
        list: Lista de nomes das classes.
    """
    model = load_model(model_path, compile=False, custom_objects={
        "BatchNormalization": BatchNormalization,
        "DepthwiseConv2D": custom_depthwise_conv2d
    })

    # Carrega os nomes das classes, removendo espaços em branco, quebras de linha e index
    with open(labels_path, "r") as file:
        class_names = [line.strip().split(' ', 1)[1] for line in file.readlines()]
    return model, class_names

def predict_image_class(model, class_names, image_array):
    """
    Faz a predição da classe da imagem.

    Args:
        model: Modelo Keras.
        class_names (list): Lista de nomes das classes.
        image_array (np.ndarray): Array numpy da imagem processada.

    Returns:
        str: Nome da classe predita.
        float: Score de confiança da predição.
    """
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = image_array
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]
    return class_name, confidence_score

def send_notification(class_name, confidence_score):
    """
    Envia uma notificação com os detalhes da predição.

    Args:
        class_name (str): Nome da classe predita.
        confidence_score (float): Score de confiança da predição.
    """
    
    class_name = class_name if confidence_score > 0.7 else "Desconhecido"

    print(f"Predição: {class_name}\nConfiança: {confidence_score:.6f}")

    if class_name == "Cachorro":
        print("Ação para Cachorro: Levar para passear.")
    elif class_name == "Gato":
        print("Ação para Gato: Dar comida.")
    else:
        print(f"Imagem não reconhecida ou confiança baixa ({confidence_score:.6f}).")
    
    print("Notificação enviada!")

def main(image_path):
    model_path = "src/models/keras_model.h5"
    labels_path = "src/models/labels.txt"

    model, class_names = load_model_and_classes(model_path, labels_path)
    image_array = load_and_preprocess_image(image_path)
    class_name, confidence_score = predict_image_class(model, class_names, image_array)

    send_notification(class_name, confidence_score)