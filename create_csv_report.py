import pandas as pd
import datetime
import os

# Load trained model
model.load_state_dict(torch.load("face_recognition_model.pth"))
model.eval()
model.to(device)

# Load Haarcascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Create an attendance CSV file if it doesn't exist
attendance_file = "attendance.csv"
if not os.path.exists(attendance_file):
    df = pd.DataFrame(columns=["Employee Name", "Time"])
    df.to_csv(attendance_file, index=False)

cap = cv2.VideoCapture(0)
recognized_names = set()  # Store recognized employees to prevent duplicate entries in a session

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (100, 100))
        face_img = np.transpose(face_img, (2, 0, 1))  # Convert to PyTorch format
        face_img = torch.tensor(face_img, dtype=torch.float32).unsqueeze(0).to(device) / 255.0
        
        with torch.no_grad():
            output = model(face_img)
            probabilities = F.softmax(output, dim=1)
            employee_id = torch.argmax(probabilities, dim=1).item()

        employee_name = class_names[employee_id] if employee_id < len(class_names) else "Unknown"
        
        cv2.putText(frame, employee_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Record attendance if employee is recognized and not already logged
        if employee_name != "Unknown" and employee_name not in recognized_names:
            recognized_names.add(employee_name)  # Prevent duplicate entries
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            df = pd.read_csv(attendance_file)
            new_entry = pd.DataFrame({"Employee Name": [employee_name], "Time": [timestamp]})
            df = pd.concat([df, new_entry], ignore_index=True)
            df.to_csv(attendance_file, index=False)
            print(f"Attendance marked for {employee_name} at {timestamp}")

    cv2_imshow(frame)  # Display in Colab

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
