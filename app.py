from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import easyocr
import csv
import uuid
import pymysql
import requests
from flask_socketio import SocketIO  # Ajoutez cette ligne pour importer SocketIO

###############################
from flask import Flask, render_template, redirect, url_for, request, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, current_user  # Import current_user
from flask_bcrypt import Bcrypt
from flask_bcrypt import check_password_hash
###############################

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

##############################################################################

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/testpfa2'
app.config['SECRET_KEY'] = 'your_secret_key'  
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

class User(db.Model, UserMixin):
    user_name = db.Column(db.String(50), primary_key=True, nullable=False)
    user_first_name = db.Column(db.String(50), nullable=False)
    user_last_name = db.Column(db.String(50), nullable=False)
    user_password = db.Column(db.String(255), nullable=False)

    def get_id(self):
        return str(self.user_name)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(str(user_id))

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_name = request.form['user_name']
        password = request.form['password']
        user = User.query.filter_by(user_name=user_name).first()
        if user and bcrypt.check_password_hash(user.user_password, password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Login failed. Check your username and password.', 'danger')
    return render_template('login.html')

@app.route("/home")
def home2():
    return render_template("home.html")   
@app.route('/dashboard')
def dashboard():
    if current_user.is_authenticated:
        return render_template("index.html")   
    #f"Welcome to the dashboard, {current_user.user_name}!"
    
    else:
        return "Please log in to access the dashboard."


################################################################################
CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
paths = {
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME),
}

# Load the detection model and label map
configs = config_util.get_configs_from_pipeline_file('Tensorflow/workspace/models/my_ssd_mobnet/pipeline.config')
detection_model = model_builder.build(model_config=configs['model'], is_training=False)
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-11')).expect_partial()
category_index = label_map_util.create_category_index_from_labelmap('Tensorflow/workspace/annotations/label_map.pbtxt')


def detect_objects(image_np):
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'] + label_id_offset,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=5,
        min_score_thresh=.8,
        agnostic_mode=False
    )

    return image_np_with_detections
# Fonction pour vérifier le numéro de plaque d'immatriculation dans la base de données
def check_license_plate_in_database(license_plate_number):
    try:
        # Établir une connexion à la base de données
        db_connection = pymysql.connect(
            host='127.0.0.1',
            user='root',
            password='',
            database='testpfa2'
        )

        # Création d'un curseur
        cursor = db_connection.cursor()

        # Exemple de requête SQL
        query = f"SELECT * FROM vehicule WHERE licenseplate = '{license_plate_number}'"
        cursor.execute(query)

        # Récupération des résultats
        results = cursor.fetchall()

        # Vérification des résultats
        if results:
            print(f"Le numéro de plaque d'immatriculation {license_plate_number} existe dans la base de données.")
        else:
            print(f"Le numéro de plaque d'immatriculation {license_plate_number} n'existe pas dans la base de données.")

    except Exception as e:
        print(f"Erreur lors de la vérification dans la base de données : {str(e)}")

    finally:
        # Fermeture du curseur et de la connexion
        cursor.close()
        db_connection.close()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


def filter_text(region, ocr_result, region_threshold):
    rectangle_size = region.shape[0] * region.shape[1]
    plate = []
    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))
        if length * height / rectangle_size > region_threshold:
            plate.append(result[1])
    return plate
detection_threshold = 0.7

region_threshold = 0.6
def ocr_it(image, detections, detection_threshold, region_threshold):
    # Scores, boxes, and classes above threshold
    scores = list(filter(lambda x: x > detection_threshold, detections['detection_scores']))
    boxes = detections['detection_boxes'][:len(scores)]

    # Full image dimensions
    width = image.shape[1]
    height = image.shape[0]

    # Apply ROI filtering and OCR
    for idx, box in enumerate(boxes):
        roi = box * [height, width, height, width]
        region = image[int(roi[0]):int(roi[2]), int(roi[1]):int(roi[3])]
        reader = easyocr.Reader(['en'])
        ocr_result = reader.readtext(region)

        text = filter_text(region, ocr_result, region_threshold)

        return text, region


def save_results(text, region, csv_filename, folder_path):
    img_name = '{}.jpg'.format(uuid.uuid1())

    cv2.imwrite(os.path.join(folder_path, img_name), region)
    with open(csv_filename, mode='a', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([img_name, text])


#################Web Socke########################"
@socketio.on('license_plate_detected')
def handle_license_plate_detection(data):
    plate_number = data.get('license_plate')

    # Check if the license plate exists in the database
    if check_license_plate_in_database(plate_number):
        # Notify Angular through WebSocket
        socketio.emit('notification', {'message': 'License plate found', 'license_plate': plate_number})



# Route pour exécuter la détection d'objets
@app.route("/detect_objects", methods=["POST"])
def detect_objects_route():
    try:
        # Capture de la vidéo depuis la webcam
        cap = cv2.VideoCapture(0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        while cap.isOpened():
            ret, frame = cap.read()
            image_np = np.array(frame)
            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
            detections = detect_fn(input_tensor)
    
            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                          for key, value in detections.items()}
            detections['num_detections'] = num_detections
           # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
            label_id_offset = 1
            image_np_with_detections = image_np.copy()

            viz_utils.visualize_boxes_and_labels_on_image_array(
                        image_np_with_detections,
                        detections['detection_boxes'],
                        detections['detection_classes']+label_id_offset,
                        detections['detection_scores'],
                        category_index,
                        use_normalized_coordinates=True,
                        max_boxes_to_draw=5,
                        min_score_thresh=.8,
                        agnostic_mode=False)
            try:
                text,region = ocr_it(image_np_with_detections,detections,detection_threshold,region_threshold)
                save_results(text,region,'realtimeresults.csv','Detection_Images')
        
          # Check if the license plate exists in the database
                if text:
                    for plate_number in text:
                        check_license_plate_in_database(plate_number)
                        socketio.emit('notification', {'message': 'License plate  wal9inahaaaaa', 'license_plate': plate_number})


            except Exception as e:
                print(f"Error during processing: {str(e)}")
            # Exécution de la détection d'objets
            image_np_with_detections = detect_objects(image_np)

            # Affichage de la vidéo avec les détections
            cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))

            if cv2.waitKey(10) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

    return jsonify({"status": "success"})


if __name__ == "__main__":
    app.run(debug=True)
