import os
from dotenv import load_dotenv
load_dotenv()
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

from flask import Flask, render_template, request, redirect, url_for, flash
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input 

app = Flask(__name__)
app.secret_key = os.getenv('MY_SECRET_KEY')
EMAIL_ADDRESS = os.getenv('EMAIL_ADDRESS')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')
RECEIVER_EMAIL = os.getenv('RECEIVER_EMAIL')

model = None

#Loading Model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_A = os.path.join(BASE_DIR, "food101_densenet201_finetune.keras")
PATH_B = os.path.abspath("food101_densenet201_finetune.keras")
PATH_C = os.path.abspath("app/food101_densenet201_finetune.keras")

if os.path.exists(PATH_A):
    MODEL_PATH = PATH_A
elif os.path.exists(PATH_B):
    MODEL_PATH = PATH_B
else: 
    MODEL_PATH = PATH_C

#Class names
class_names = ['apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito', 'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake', 'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla', 'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder', 'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes', 'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras', 'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice', 'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna', 'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup', 'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck', 'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake', 'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles']

IMG_SIZE = (224, 224)
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def predict_image(img_path):
    global model 
    
    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    img = tf.expand_dims(img, axis=0)

    predictions = model.predict(img)[0]
    sorted_indices = np.argsort(predictions)[::-1]

    top_5_predictions = []
    for i in range(5):
        idx = sorted_indices[i]
        class_name = class_names[idx]
        confidence = float(predictions[idx])
        top_5_predictions.append({
            'class': class_name,
            'confidence': round(confidence*100,2)
        })

    predicted_class_idx = sorted_indices[0]
    predicted_class = class_names[predicted_class_idx]
    predicted_confidence = float(predictions[predicted_class_idx])

    return predicted_class, predicted_confidence, top_5_predictions

@app.route('/', methods = ['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
            
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
            
        if file:
            try:
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)
                predicted_class, predicted_confidence, top_5 = predict_image(file_path)
                formatted_confidence = round(predicted_confidence * 100, 2)
                web_accessible_img_path = f"static/uploads/{file.filename}"
                return render_template(
                    'index.html',
                    prediction=predicted_class,
                    confidence=formatted_confidence,
                    top_5=top_5,
                    image_path=web_accessible_img_path
                )
            except Exception as e:
                print(f"ERROR: {str(e)}")
                return f"Prediction failed. Error log: {str(e)}", 500
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
