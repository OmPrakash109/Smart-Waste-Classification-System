from ultralytics import YOLO
import time
import streamlit as st
import cv2
import settings
import threading
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import logging
import os

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def load_model(model_path):
    model = YOLO(model_path)
    return model

def classify_waste_type(detected_items):
    recyclable_items = set(detected_items) & set(settings.RECYCLABLE)
    non_recyclable_items = set(detected_items) & set(settings.NON_RECYCLABLE)
    hazardous_items = set(detected_items) & set(settings.HAZARDOUS)
    
    return recyclable_items, non_recyclable_items, hazardous_items

def remove_dash_from_class_name(class_name):
    return class_name.replace("_", " ")

class VideoProcessor(VideoProcessorBase):
    def __init__(self, model):
        self.model = model
        self.unique_classes = set()
        self.last_detection_time = 0

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            img = cv2.resize(img, (640, int(640*(9/16))))
            
            res = self.model.predict(img, conf=0.6)
            names = self.model.names
            detected_items = set()

            for result in res:
                new_classes = set([names[int(c)] for c in result.boxes.cls])
                if new_classes != self.unique_classes:
                    self.unique_classes = new_classes
                    detected_items.update(self.unique_classes)
                    
                    # Update session state from a thread-safe context
                    if detected_items:
                        recyclable_items, non_recyclable_items, hazardous_items = classify_waste_type(detected_items)
                        st.session_state['detected_items'] = {
                            'recyclable': recyclable_items,
                            'non_recyclable': non_recyclable_items,
                            'hazardous': hazardous_items
                        }
                        st.session_state['update_ui'] = True

            return av.VideoFrame.from_ndarray(res[0].plot(), format="bgr24")
        except Exception as e:
            logger.error(f"Error in video processing: {str(e)}")
            return frame

def play_webcam(model):
    # Initialize session state variables
    if 'detected_items' not in st.session_state:
        st.session_state['detected_items'] = {'recyclable': set(), 'non_recyclable': set(), 'hazardous': set()}
    if 'update_ui' not in st.session_state:
        st.session_state['update_ui'] = False
    
    # Create placeholders
    recyclable_placeholder = st.sidebar.empty()
    non_recyclable_placeholder = st.sidebar.empty()
    hazardous_placeholder = st.sidebar.empty()
    
    # Configure WebRTC with multiple STUN servers and TURN servers
    # Using a combination of free TURN servers for better reliability
    rtc_configuration = RTCConfiguration(
        {"iceServers": [
            {"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]},
            {
                "urls": [
                    "turn:openrelay.metered.ca:80",
                    "turn:openrelay.metered.ca:443",
                    "turn:openrelay.metered.ca:443?transport=tcp"
                ],
                "username": "openrelayproject",
                "credential": "openrelayproject"
            },
            {
                "urls": [
                    "turn:global.turn.twilio.com:3478?transport=udp",
                    "turn:global.turn.twilio.com:3478?transport=tcp",
                    "turn:global.turn.twilio.com:443?transport=tcp"
                ],
                "username": "f4b4035eaa76f4a55de5f4351567653ee4ff6fa97b50b6b334fcc1be9c27212d",
                "credential": "myL7UCqLYXcr1zBzJ7+IZDJloLe4wG4UvLX8H3Kv3lY="
            }
        ]}
    )
    
    # Create WebRTC streamer with additional options for better stability
    webrtc_ctx = webrtc_streamer(
        key="waste-detection",
        video_processor_factory=lambda: VideoProcessor(model),
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        video_html_attrs={"controls": True, "autoPlay": True, "style": {"width": "100%", "height": "auto"}},
        async_transform=True,
    )
    
    # Update UI based on detections
    if webrtc_ctx.state.playing:
        if st.session_state.get('update_ui', False):
            items = st.session_state['detected_items']
            
            if items['recyclable']:
                detected_items_str = "\n- ".join(remove_dash_from_class_name(item) for item in items['recyclable'])
                recyclable_placeholder.markdown(
                    f"<div class='stRecyclable'>Recyclable items:\n\n- {detected_items_str}</div>",
                    unsafe_allow_html=True
                )
            else:
                recyclable_placeholder.empty()
                
            if items['non_recyclable']:
                detected_items_str = "\n- ".join(remove_dash_from_class_name(item) for item in items['non_recyclable'])
                non_recyclable_placeholder.markdown(
                    f"<div class='stNonRecyclable'>Non-Recyclable items:\n\n- {detected_items_str}</div>",
                    unsafe_allow_html=True
                )
            else:
                non_recyclable_placeholder.empty()
                
            if items['hazardous']:
                detected_items_str = "\n- ".join(remove_dash_from_class_name(item) for item in items['hazardous'])
                hazardous_placeholder.markdown(
                    f"<div class='stHazardous'>Hazardous items:\n\n- {detected_items_str}</div>",
                    unsafe_allow_html=True
                )
            else:
                hazardous_placeholder.empty()
                
            st.session_state['update_ui'] = False
    else:
        # Clear the placeholders when not streaming
        recyclable_placeholder.empty()
        non_recyclable_placeholder.empty()
        hazardous_placeholder.empty()
