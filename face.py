import cv2
from deepface import DeepFace

# Initialize the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Failed to capture image.")
        break

    # Convert frame to RGB (OpenCV uses BGR by default)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    try:
        # Analyze the frame for face detection and emotion recognition
        analysis = DeepFace.analyze(rgb_frame, actions=['emotion'])
        
        # Extract the emotion information
        dominant_emotion = analysis['dominant_emotion']
        
        # Draw rectangle around detected face
        for face in analysis['region']:
            x, y, w, h = face['x'], face['y'], face['w'], face['h']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # Display the dominant emotion on the frame
            cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    except Exception as e:
        print("No face detected or error in analysis:", str(e))

    # Display the resulting frame
    cv2.imshow('Emotion Recognition', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()
