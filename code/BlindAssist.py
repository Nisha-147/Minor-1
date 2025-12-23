import cv2
from gtts import gTTS  # For better multilingual TTS (supports Hindi)
import os  # For playing audio files
import easyocr
import speech_recognition as sr  # For voice commands
from ultralytics import YOLO

# Initialize speech recognizer for voice commands
recognizer = sr.Recognizer()
mic = sr.Microphone()

# Language options (bilingual: English and Hindi)
languages = {'en': 'English', 'hi': 'Hindi'}
current_lang = 'en'  # Default to English

# Initialize EasyOCR reader (will update dynamically)
reader = easyocr.Reader(['en'])  # Start with English

# Load YOLOv8 Nano model
model_path = 'yolov8n.pt'  # Or 'best.pt' for custom
model = YOLO(model_path)

# Function to narrate text using gTTS (supports Hindi)
def narrate_text(text, lang):
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save("temp_audio.mp3")
    os.system("start temp_audio.mp3" if os.name == 'nt' else "mpg321 temp_audio.mp3" if os.name == 'posix' else "afplay temp_audio.mp3")  # Play audio (adjust for your OS)
    os.remove("temp_audio.mp3")  # Clean up

# Function to detect objects in a frame
def detect_objects(frame):
    results = model(frame, verbose=False)
    detections = []
    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2, conf, cls = box.data[0].tolist()
                label = model.names[int(cls)]
                detections.append({
                    'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2,
                    'confidence': conf, 'name': label
                })
    return detections

# Function to narrate detections (bilingual: English/Hindi)
def narrate_detection(detections, lang):
    for det in detections:
        if det['confidence'] > 0.5:
            label = det['name']
            confidence = det['confidence']
            box_area = (det['xmax'] - det['xmin']) * (det['ymax'] - det['ymin'])
            if lang == 'hi':
                distance = "नज़दीक" if box_area > 50000 else "दूर"
                message = f"{label} पता चला, {distance}, विश्वास {confidence:.2f}"
            else:
                distance = "near" if box_area > 50000 else "far"
                message = f"{label} detected, {distance}, confidence {confidence:.2f}"
            narrate_text(message, lang)

# Function to read text via EasyOCR (bilingual: English/Hindi)
def read_text(frame, lang):
    global reader
    if lang != reader.lang_list[0]:  # Update reader if language changed
        reader = easyocr.Reader([lang])
    results = reader.readtext(frame)
    for bbox, text, conf in results:
        if text.strip():
            prefix = "पाठ पता चला: " if lang == 'hi' else "Text detected: "
            narrate_text(f"{prefix}{text}", lang)

# Function to listen for voice commands
def listen_for_command():
    try:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=1)
        command = recognizer.recognize_google(audio).lower()
        return command
    except:
        return None

# Main setup: Select language at start
print("Select language: 'en' for English, 'hi' for Hindi")
lang_input = input("Enter 'en' or 'hi': ").strip().lower()
if lang_input in languages:
    current_lang = lang_input
else:
    print("Invalid, defaulting to English.")

# Main loop for continuous detection
cap = cv2.VideoCapture(0)
frame_count = 0
paused = False

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Check for keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        paused = not paused
        status = "रोका गया" if current_lang == 'hi' else "Paused"
        print(f"{status}: {paused}")
        narrate_text(status, current_lang)
    elif key == ord('l'):
        current_lang = 'hi' if current_lang == 'en' else 'en'
        switch_msg = f"भाषा बदलकर {languages[current_lang]} की गई" if current_lang == 'hi' else f"Language switched to {languages[current_lang]}"
        print(switch_msg)
        narrate_text(switch_msg, current_lang)
    
    # Listen for voice commands periodically
    if frame_count % 100 == 0:  # Every ~3-4 seconds
        command = listen_for_command()
        if command:
            if 'pause' in command or 'रोक' in command:
                paused = True
            elif 'resume' in command or 'शुरू' in command:
                paused = False
            elif 'switch' in command and 'language' in command:
                current_lang = 'hi' if current_lang == 'en' else 'en'
                switch_msg = f"भाषा बदलकर {languages[current_lang]} की गई" if current_lang == 'hi' else f"Language switched to {languages[current_lang]}"
                narrate_text(switch_msg, current_lang)
    
    if not paused:
        # Run detection and narration
        detections = detect_objects(frame)
        if frame_count % 30 == 0:
            narrate_detection(detections, current_lang)
        
        # Read text periodically
        if frame_count % 150 == 0:
            read_text(frame, current_lang)
    
    # Display frame
    cv2.imshow('Frame', frame)
    frame_count += 1
    

cap.release()
cv2.destroyAllWindows()
