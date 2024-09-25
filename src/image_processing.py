import numpy as np
from PIL import Image, ImageOps

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    Carrega uma imagem, converte para RGB, redimensiona e normaliza.

    Args:
        image_path (str): Caminho para a imagem.
        target_size (tuple): Tamanho alvo para redimensionamento.

    Returns:
        np.ndarray: Imagem processada como um array numpy.
    """
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    return normalized_image_array