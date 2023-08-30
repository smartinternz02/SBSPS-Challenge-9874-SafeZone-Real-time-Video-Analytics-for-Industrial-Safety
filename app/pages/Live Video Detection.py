import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import tensorflow as tf
import numpy as np
import time
import random
import object_detection
from functools import partial
import threading
import base64
from twilio.rest import Client
from object_detection.utils import visualization_utils as viz_utils


st.markdown("# Real time Video Detection")
st.sidebar.header("Real time Video Detection")
st.divider()


run_detection = False


TWILIO_ACCOUNT_SID = "twilio_account_sid"
TWILIO_AUTH_TOKEN = "twilio_auth_token"
TWILIO_PHONE_NUMBER = "twilio_phone_number"
RECIPIENT_PHONE_NUMBER = "recipient_phone_number"
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)


message_sent = False  
session_state_lock = threading.Lock()
if "warning_message" not in st.session_state:
    st.session_state.warning_message = ""


def send_sms(message):
    global message_sent
    if not message_sent:
        try:
            twilio_client.messages.create(
                body=message,
                from_=TWILIO_PHONE_NUMBER,
                to=RECIPIENT_PHONE_NUMBER
            )
            print("SMS sent successfully!")
            message_sent = True  # Set the flag to True after sending the message
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

def send_warning(message):
    with session_state_lock:
        st.session_state.warning_message = message

def live_video_detection(detect_fns):
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

    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    while cap.isOpened(): 
        ret, frame = cap.read()
        image_np = np.array(frame)
    
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.uint8)

        for model_info in detect_fns:
            model_name = model_info["name"]
            detect_fn = model_info["detect_fn"]
            detections = detect_fn(input_tensor)
    
            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
            detections['num_detections'] = num_detections

            # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        
            category_index = model_category_indexes.get(model_name, {})

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

        for model in selected_models:
            stop_audio_button = st.button("Stop Audio")
            model_name = model['name']
            if model_name == "Fire Detection" and any(class_id == 1 for class_id in detections['detection_classes']) or any(class_id == 2 for class_id in detections['detection_classes']):
                send_warning("⚠️ Fire Detected!")
                autoplay_audio("siren.mp3")
                send_sms("⚠️ Fire detected in the industry")
                if stop_audio_button:
                    stop_audio()
            if model_name == "PPE Detection":
                if any(class_id == 2 for class_id in detections['detection_classes']):
                    st.warning("⚠️ Person Detected with NO-Safety Vest")
                if any(class_id == 3 for class_id in detections['detection_classes']):
                    st.warning("⚠️ Person Detected with NO-Gloves")
                if any(class_id == 4 for class_id in detections['detection_classes']):
                    st.warning("⚠️ Person Detected with NO-Hard Hat")
                if any(class_id == 5 for class_id in detections['detection_classes']):
                    st.warning("⚠️ Person Detected with NO-Safety Boots")
            if model_name == "Cigarette Detection" and any(class_id == 1 for class_id in detections['detection_classes']):
                st.warning("⚠️ Person Detected Smoking")
            if model_name == "Spill Detection" and any(class_id == 1 for class_id in detections['detection_classes']):
                st.warning("⚠️ Spill Detected!")

        cv2.imshow('Object Detection',  cv2.resize(image_np_with_detections, (800, 600)))

        if cv2.waitKey(10) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break


st.markdown("**Choose the Models for Detection**")
fire_detection = st.checkbox("Fire Detection")
ppe_detection = st.checkbox("PPE Detection")
cigarette_detection = st.checkbox("Cigarette Detection")
spill_detection = st.checkbox("Spill Detection")


if st.button("Start Detection"):
    run_detection = True

if st.button("Stop Detection"):
    run_detection = False

selected_models = [] 

if run_detection:   
    status_text = st.empty()

    # List of texts to display
    status_messages = [
                    "Initializing...",
                    "Loading models...",
                    "Starting video stream...",
                    "Adjusting camera settings...",
                    "Calibrating detectors...",
                    "Preparing for detection..."
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

    status_text.success("Models Loaded Successfully!")

    st.sidebar.write("Selected Models:")
    for model in selected_models:
        st.sidebar.markdown(f"- {model['name']}")

    if len(selected_models) > 0:
        thread = threading.Thread(target=live_video_detection, args=(selected_models,))
        thread.start()