import streamlit as st
import numpy as np
import object_detection
import time
import random
import tensorflow as tf
from PIL import Image
from object_detection.utils import visualization_utils as viz_utils
import base64
from twilio.rest import Client


st.markdown("# Image Detection")
st.sidebar.header("Image Detection")
st.divider()
st.markdown("Detect potential safety hazards and anomalies in static images using advanced object detection models. Choose from a variety of pre-trained models or even create custom models tailored to your industry needs.")
uploaded_file = st.file_uploader(label="Upload image here 👇", type=["jpg", "png", "jpeg"],label_visibility="hidden")


TWILIO_ACCOUNT_SID = "twilio_account_sid"
TWILIO_AUTH_TOKEN = "twilio_auth_token"
TWILIO_PHONE_NUMBER = "twilio_phone_number"
RECIPIENT_PHONE_NUMBER = "recipent_phone_number"
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)


selected_models = []  


def send_sms(message):
    try:
        twilio_client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=RECIPIENT_PHONE_NUMBER
        )
        print("SMS sent successfully!")
    except Exception as e:
        print("Error sending SMS:", str(e))

def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio autoplay="true" loop="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(
            md,
            unsafe_allow_html=True
        )
        # time.sleep(30)  

def stop_audio():
    st.audio("", format="audio/mp3")

def image_detection(uploaded_file, detect_fns):
    model_category_indexes = {
        "Fire Detection": {
            1: {'id': 1, 'name': 'fire'},
            2: {'id': 2, 'name': 'firew'},
            3: {'id': 3, 'name': 'smoke'}
        },
        "PPE Detection": {
            1: {'id': 1, 'name': 'Person'},
            2: {'id': 2, 'name': 'NO-Safety Vest'},
            3: {'id': 3, 'name': 'NO-Gloves'},
            4: {'id': 4, 'name': 'NO-Hardhat'},
            5: {'id': 5, 'name': 'NO-Safety Boot'}
        },
        "Cigarette Detection": {
            1: {'id': 1, 'name': 'Cigarette'},
            2: {'id': 2, 'name': 'Person'}
        },
        "Spill Detection": {
            1: {'id': 1, 'name': 'Spill'}
        }
    }

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Prepare an image for visualization with all detections
    image_np_with_detections = np.array(image.copy())

    # Perform object detection for each model
    for model_info in detect_fns:
        model_name = model_info["name"]
        detect_fn = model_info["detect_fn"]

        input_tensor = tf.convert_to_tensor(np.array(image))
        input_tensor = input_tensor[tf.newaxis, ...]

        detections = detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        category_index = model_category_indexes.get(model_name, {})

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.4,  # Adjust this value as needed
            agnostic_mode=False)

    st.image(image_np_with_detections, caption="Object Detection Result", use_column_width=True)

    for model in selected_models:
        stop_audio_button = st.button("Stop Audio")
        model_name = model['name']
        if model_name == "Fire Detection" and any(class_id == 1 for class_id in detections['detection_classes']):
            st.warning("Fire Detected!", icon="⚠️")
            autoplay_audio("siren.mp3")
            send_sms("⚠️ Fire detected in the industry")
            if stop_audio_button:
                stop_audio()
        if model_name == "PPE Detection":
            if any(class_id == 2 for class_id in detections['detection_classes']):
                st.warning("Person Detected with NO-Safety Vest",icon="⚠️")
            if any(class_id == 3 for class_id in detections['detection_classes']):
                st.warning("Person Detected with NO-Gloves",icon="⚠️")
            if any(class_id == 4 for class_id in detections['detection_classes']):
                st.warning("Person Detected with NO-Hard Hat",icon="⚠️")
            if any(class_id == 5 for class_id in detections['detection_classes']):
                st.warning("Person Detected with NO-Safety Boots",icon="⚠️")
        if model_name == "Cigarette Detection" and any(class_id == 1 for class_id in detections['detection_classes']):
            st.warning("Person Detected Smoking", icon="⚠️")
        if model_name == "Spill Detection" and any(class_id == 1 for class_id in detections['detection_classes']):
            st.warning("Spill Detected!", icon="⚠️")


st.markdown("**Choose the Models for Detection**")
fire_detection = st.checkbox("Fire Detection")
ppe_detection = st.checkbox("PPE Detection")
cigarette_detection = st.checkbox("Cigarette Detection")
spill_detection = st.checkbox("Spill Detection")


generate_model_button = st.button("Generate Model")


if generate_model_button:
    status_text = st.empty()

    # List of texts to display
    status_messages = [
        "Collecting the Images...",
        "Preprocessing the collected data...",
        "Generating the Model...",
        "Detecting Objects...",
        "Analyzing Results..."
    ]

    for message in status_messages:
        status_text.info(message)
        time.sleep(random.uniform(1, 3))  # Random sleep between 1 and 3 seconds
    if fire_detection:
        selected_models.append({"name": "Fire Detection", "detect_fn": tf.saved_model.load(r'C:\Users\Hariprasath\Downloads\saved_models\fire_saved_model')})
    if ppe_detection:
        selected_models.append({"name": "PPE Detection", "detect_fn": tf.saved_model.load(r'C:\Users\Hariprasath\Downloads\saved_models\construction_saved_model')})
    if cigarette_detection:
        selected_models.append({"name": "Cigarette Detection", "detect_fn": tf.saved_model.load(r'C:\Users\Hariprasath\Downloads\saved_models\smoke_saved_model')})
    if spill_detection:
        selected_models.append({"name": "Spill Detection", "detect_fn": tf.saved_model.load(r'C:\Users\Hariprasath\Downloads\saved_models\spill_saved_model')})

    status_text.success("Detection Completed!")
    st.sidebar.write("Selected Models:")
    for model in selected_models:
        st.sidebar.markdown(f"- {model['name']}")

if uploaded_file is not None:
    image_detection(uploaded_file, selected_models)