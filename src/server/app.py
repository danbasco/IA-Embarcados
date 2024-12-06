from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os, numpy as np
import cv2 as cv

class App(Flask):
    def __init__(self, model):
        super().__init__(__name__)

        self.model = model

        UPLOAD_FOLDER = 'images'
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        self.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

        self.add_url_rule('/', 'home', self.home)
        self.add_url_rule('/upload', 'upload_image', self.upload_image, methods=['POST'])
        self.add_url_rule('/number', 'get_number', self.get_number, methods=['GET'])

        self.result = None

    def home(self):
        return "Sistemas Embarcados, para método POST visitar /upload"

    def upload_image(self):

        if 'image' not in request.files:
            return jsonify({'error': 'Nenhum arquivo encontrado no campo "image"'}), 400

        file = request.files['image']

        if file.filename == '':
            return jsonify({'error': 'Nenhum arquivo selecionado'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(self.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        result = self.process_image(filepath)

        self.result = self.model.test(
            self.getimage("images/img1.png")
        )

        return jsonify({'message': 'Imagem processada com sucesso', 'result': result}), 200

    def get_number(self):
        if self.result is None:
            return jsonify({'error': 'Nenhum resultado encontrado, envie uma imagem primeiro'}), 400
        return jsonify({'message': '', 'result': self.result})

    def process_image(self, filepath):

        image = cv.imread(filepath)
        dimensions = image.shape
        return {'height': dimensions[0], 'width': dimensions[1], 'channels': dimensions[2]}

    def getimage(self, dir: str) -> np.ndarray:

        img = cv.imread(dir, cv.IMREAD_GRAYSCALE)
        img = cv.resize(img, (28, 28))
        img = img.flatten() / 255.0
        img = np.expand_dims(img, axis=-1)
        return np.expand_dims(img, axis=0)

