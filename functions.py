def capture_images(person_name):
    import cv2
    import os
    import face_recognition
    # Path to the datasets folder
    datasets_folder = 'datasets'

    # Create a new folder for the person in the datasets folder
    person_folder = os.path.join(datasets_folder, person_name)
    if not os.path.isdir(person_folder):
        os.makedirs(person_folder)

    # Open the video capture device
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Set the image counter to 0
    img_counter = 0

    # Loop over the video feed
    while True:
        # Capture a frame from the video feed
        ret, frame = cap.read()

        # If the frame was successfully captured
        if ret:
            # Convert the image to RGB format
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect the face in the image
            face_locations = face_recognition.face_locations(rgb_frame)
            if not face_locations:
                # No face detected
                continue

            # Display the frame
            cv2.imshow("Capturing Images", frame)

            # Wait for the 's' key to be pressed to capture an image
            k = cv2.waitKey(1)
            if k == ord('s'):
                # Increment the image counter
                img_counter += 1

                # Set the image filename
                img_name = os.path.join(person_folder, '{}.png'.format(img_counter))

                # Save the image
                cv2.imwrite(img_name, frame)

                # Display a message indicating that the image was saved
                print("{} saved!".format(img_name))

            # If the 'q' key is pressed, exit the loop
            elif k == ord('q'):
                break

    # Release the video capture device and close all windows
    cap.release()
    cv2.destroyAllWindows()

def train_facial_recognition_model(dataset_path, model_path):
    import os
    import face_recognition
    import pickle
    # Load the dataset of images for each person
    known_faces = []
    known_names = []

    for person in os.listdir(dataset_path):
        if not person.startswith('.'):
            person_path = os.path.join(dataset_path, person)
            for image_file in os.listdir(person_path):
                if not image_file.startswith('.'):
                    image_path = os.path.join(person_path, image_file)
                    image = face_recognition.load_image_file(image_path)
                    face_encodings = face_recognition.face_encodings(image)
                    if len(face_encodings) > 0:
                        face_encoding = face_encodings[0]
                        known_faces.append(face_encoding)
                        known_names.append(person)

    # Train the facial recognition model
    face_recognizer = {"known_faces": known_faces, "known_names": known_names}

    # Save the trained model to disk
    with open(model_path, "wb") as f:
        f.write(pickle.dumps(face_recognizer))

def smart_attendance_system():
    import datetime
    import csv
    import cv2
    import face_recognition
    import pickle
    import os
    import mysql.connector
    # Load the face detection classifier and trained facial recognition model
    face_recognizer = pickle.load(open("model.pkl", "rb"))
    known_face_encodings = face_recognizer["known_faces"]
    known_face_names = face_recognizer["known_names"]

    # Set up the file name
    today = datetime.datetime.today().strftime('%Y_%m_%d')
    TableName = f"attendance_{today}"
    filename = f"attendance_{today}.csv"

    # Check if the file exists
    file_exists = os.path.isfile(filename)

    # Open the CSV file in append mode
    with open(filename, mode='a', newline='') as file:
        # Initialize the CSV writer
        writer = csv.writer(file)

        # Connect to the MySQL database
        mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            password="12345678",
            database="Student_Attendance"
        )
        mycursor = mydb.cursor()

        mycursor.execute(
            f"CREATE TABLE IF NOT EXISTS {TableName} (date DATE, time TIME, student_name VARCHAR(255), status VARCHAR(255))")

        # Write the header row if the file is empty
        if not file_exists:
            writer.writerow(['Date', 'Time', 'Student Name', 'Status'])

        # Start the webcam
        video_capture = cv2.VideoCapture(0)
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Check if the video capture is working
        if not video_capture.isOpened():
            raise Exception("Could not open video capture")

        attendance_recorded = set()

        while True:
            # Capture a frame from the webcam
            ret, frame = video_capture.read()

            # Check if the frame is None
            if frame is None:
                raise Exception("Could not read frame from video capture")

            # Convert the frame to RGB
            rgb_frame = frame[:, :, ::-1]

            # Detect faces in the frame
            # Find all the faces in the frame
            face_locations = face_recognition.face_locations(rgb_frame)

            # Get the face encoding for each face in the frame
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            # Loop through each face in the frame
            for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
                # Try to match the face to a known face
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)

                # If a match was found, get the name of the person
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]

                    # Display the name on the frame
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

                    # Save the attendance record in the CSV file and MySQL database if it hasn't already been recorded
                    if name not in attendance_recorded:
                        date = str(datetime.datetime.now().date())
                        time = str(datetime.datetime.now().time())
                        student_name = name
                        status = "Present"
                        writer.writerow([date, time, student_name, status])
                        attendance_recorded.add(name)

                        # Save the attendance record in the MySQL database
                        sql = f"INSERT INTO {TableName} (date, time, student_name, status) VALUES (%s, %s, %s, %s)"
                        val = (date, time, student_name, status)
                        mycursor.execute(sql, val)
                        mydb.commit()

                else:
                    # Display "Unknown" on the frame
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, "Unknown", (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            # Display the frame
            cv2.imshow("Smart Attendance System", frame)

            # Break the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the webcam and destroy the window
        video_capture.release()
        cv2.destroyAllWindows()



