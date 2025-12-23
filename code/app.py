import streamlit as st
import cv2
from ultralytics import YOLO
import pyttsx3

# Initialize YOLO model
model = YOLO("best.pt")

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Narration function
def narrate(text, lang='en'):
    if text.strip():
        voices = engine.getProperty('voices')
        if lang == 'hi':
            # Try to pick Hindi voice if available
            for v in voices:
                if "hi" in v.id.lower() or "hindi" in v.name.lower():
                    engine.setProperty('voice', v.id)
                    break
        else:
            engine.setProperty('voice', voices[0].id)  # default English
        engine.say(text)
        engine.runAndWait()

# Streamlit UI
st.title("SaArthi Vision - Object Detection")

# Language selection box
language_choice = st.selectbox("Select Narration Language", ["English", "Hindi"])
current_lang = 'en' if language_choice == "English" else 'hi'

run_detection = st.checkbox("Run Object Detection")
camera_index = st.number_input("Camera Index", min_value=0, value=0)

if run_detection:
    cap = cv2.VideoCapture(camera_index)
    detected_set = set()
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Camera not available.")
            break

        results = model(frame, verbose=False)
        names = set()

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if conf < 0.5:
                    continue
                name = model.names[cls_id]
                names.add(name)

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        new_items = names - detected_set
        if new_items:
            if current_lang == 'en':
                text = "In front of you: " + ", ".join(new_items)
            else:
                text = "आपके सामने है " + ", ".join(new_items)
            st.write(text)
            narrate(text, lang=current_lang)
            detected_set.update(new_items)

        stframe.image(frame, channels="BGR")

    cap.release()