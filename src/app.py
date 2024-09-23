import os
import uuid
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

app = Flask(__name__)

# Configurar la carpeta para subir imágenes
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'src', 'descargas')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Verificar extensiones de archivo permitidas
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Cargar el modelo
MODEL_PATH = os.path.join(os.getcwd(), 'model_transfer.keras')
model = load_model(MODEL_PATH)

# Ruta principal
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Ruta para la predicción
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        # Guardar la imagen con un nombre único y su extensión original
        extension = file.filename.rsplit('.', 1)[1].lower()
        filename = str(uuid.uuid4()) + "." + extension
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)  # Guardar la imagen en la carpeta 'descargas'

        # Procesar la imagen para la predicción
        img = Image.open(filepath)
        img = img.resize((100, 100))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0) / 255.0

        # Realizar la predicción con el modelo
        pred = model.predict(img)
        pred_class = 'Perro' if pred > 0.5 else 'Gato'

        # Mostrar el resultado junto con la imagen subida
        return render_template('result.html', prediction=pred_class, image_filename=filename)

# Ruta para servir las imágenes
@app.route('/descargas/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)