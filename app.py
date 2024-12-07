import os
import numpy as np
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

# Configure upload directory
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Load the pre-trained model
model = load_model('model.h5')

# Set the image size to what your model expects
IMG_WIDTH, IMG_HEIGHT = 128, 128  # Adjust if your model expects a different size

# Define classes in the order the model was trained.
# Replace these with the actual class names your model predicts:
classes = ["CORN_healthy", "APPLE_healthy", "CORN_Notherm_Leaf_Blight","APPLE_Black_Rot","CORN_Cercospora_Leaf_Spot","APPLE_Cedar_Rust","CORN_Common_Rust"]  # Example placeholder classes

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    # Load the image
    img = load_img(image_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
    # Convert to array
    img_array = img_to_array(img) / 255.0  # normalize if the model was trained on normalized images
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'photo' not in request.files:
            return "No file part in the request."

        file = request.files['photo']
        if file.filename == '':
            return "No file selected."

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Preprocess the image
            img = preprocess_image(filepath)

            # Predict
            preds = model.predict(img)
            print("Predictions:", preds)  # Debug print
            print("Shape of preds:", preds.shape)  # Debug print

            # Assuming preds is of shape (1, number_of_classes)
            predicted_class_index = np.argmax(preds[0])
            print("Predicted class index:", predicted_class_index)  # Debug print
            print("Number of classes:", len(classes))  # Debug print

            if 0 <= predicted_class_index < len(classes):
                predicted_label = classes[predicted_class_index]
            else:
                predicted_label = "Unknown"

            file_url = url_for('static', filename='uploads/' + filename, _external=True)
            return render_template('index.html',
                                   message="File uploaded and evaluated successfully!",
                                   file_url=file_url,
                                   prediction=predicted_label)

        return "File type not allowed. Allowed types: png, jpg, jpeg, gif."

    # On GET, just show the form
    return render_template('index.html')

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
