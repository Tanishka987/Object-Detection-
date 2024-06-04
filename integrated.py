import numpy as np
import cv2
from scipy.spatial import distance as dist
import pyttsx3

# Initialize the TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed percent (can go over 100)
engine.setProperty('volume', 0.9)  # Volume 0-1
# Constants for distance measurement
Known_distance = 30  # Inches
Known_width = 5.7  # Inches
Known_widths = {
    'person': 16,  # Inches
    'car': 72,  # Inches
    'bicycle': 20,  # Inches
    'bed': 60,          # Inches
    'couch': 80,        # Inches
    'chair': 20,        # Inches
    'mirror': 24,       # Inches
    'dining table': 60, # Inches
    'window': 48,       # Inches
    'desk': 48,         # Inches
    'toilet': 16,       # Inches
    'door': 36,         # Inches
    'tv': 45,           # Inches
    'laptop': 13,       # Inches
    'mouse': 2.5,       # Inches
    'remote': 2,        # Inches
    'book': 6,          # Inches
    'scissors': 3,      # Inches
    'teddy bear': 10,
    'cell phone ':2.5,
}
thres = 0.5  # Threshold to detect object
nms_threshold = 0.2  # (0.1 to 1) 1 means no suppress, 0.1 means high suppress

# Colors in BGR Format
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLACK = (0, 0, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
ORANGE = (0, 69, 255)

font = cv2.FONT_HERSHEY_PLAIN
fonts = cv2.FONT_HERSHEY_COMPLEX

# Load class names
classNames = []
with open('coco.names', 'r') as f:
    classNames = f.read().splitlines()

Colors = np.random.uniform(0, 255, size=(len(classNames), 3))

weightsPath = "frozen_inference_graph.pb"
configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"

face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Function to find focal length
def FocalLength(measured_distance, real_width, width_in_rf_image):
    return (width_in_rf_image * measured_distance) / real_width

# Function to estimate distance
def Distance_finder(Focal_Length, real_face_width, face_width_in_frame):
    return (real_face_width * Focal_Length) / face_width_in_frame

# Function to detect faces and draw rectangles
def face_data(image, CallOut, Distance_level):
    face_width = 0
    face_center_x = 0
    face_center_y = 0
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_image, 1.3, 5)
    for (x, y, w, h) in faces:
        LLV = int(h * 0.12)
        cv2.line(image, (x, y + LLV), (x + w, y + LLV), GREEN, 2)
        cv2.line(image, (x, y + h), (x + w, y + h), GREEN, 2)
        cv2.line(image, (x, y + LLV), (x, y + LLV + LLV), GREEN, 2)
        cv2.line(image, (x + w, y + LLV), (x + w, y + LLV + LLV), GREEN, 2)
        cv2.line(image, (x, y + h), (x, y + h - LLV), GREEN, 2)
        cv2.line(image, (x + w, y + h), (x + w, y + h - LLV), GREEN, 2)
        face_width = w
        face_center_x = int(w / 2) + x
        face_center_y = int(h / 2) + y
        if Distance_level < 10:
            Distance_level = 10
        if CallOut:
            cv2.line(image, (x, y - 11), (x + 180, y - 11), ORANGE, 28)
            cv2.line(image, (x, y - 11), (x + 180, y - 11), YELLOW, 20)
            cv2.line(image, (x, y - 11), (x + Distance_level, y - 11), GREEN, 18)
    return face_width, faces, face_center_x, face_center_y

# Read reference image to calculate focal length
ref_image = cv2.imread("lena.png")
ref_image_face_width, _, _, _ = face_data(ref_image, False, 0)
Focal_length_found = FocalLength(Known_distance, Known_width, ref_image_face_width)
print(Focal_length_found)

def make_chunks(EdgeArray, size_of_chunk):
    chunks = []
    for i in range(0, len(EdgeArray), size_of_chunk):
        chunks.append(np.sum(EdgeArray[i:i + size_of_chunk]))
    return chunks

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    StepSize = 5

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Face and object detection
        classIds, confs, bbox = net.detect(frame, confThreshold=thres)
        bbox = list(bbox)
        confs = list(np.array(confs).reshape(1, -1)[0])
        confs = list(map(float, confs))
        indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

        face_width_in_frame, Faces, FC_X, FC_Y = face_data(frame, True, 0)
        if len(classIds) != 0:
            for i in indices:
                box = bbox[i]
                confidence = str(round(confs[i], 2))
                label = classNames[classIds[i] - 1]
                color = Colors[classIds[i] - 1]
                x, y, w, h = box[0], box[1], box[2], box[3]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness=2)
                cv2.putText(frame, classNames[classIds[i] - 1] + " " + confidence, (x + 10, y + 20),
                            font, 1, color, 2)
                if label in Known_widths:
                    object_width = Known_widths[label]
                    distance = Distance_finder(Focal_length_found, object_width, w)
                    distance = round(distance, 2)
                    cv2.putText(frame, f"{distance} Inches", (x, y - 10), font, 1, color, 2)
                    if distance < 25:  # For example, move away warning if the object is too close
                        engine.say(f"{label} is close")
                        engine.runAndWait()

        for (face_x, face_y, face_w, face_h) in Faces:
            if face_width_in_frame != 0:
                Distance = Distance_finder(Focal_length_found, Known_width, face_width_in_frame)
                Distance = round(Distance, 2)
                Distance_level = int(Distance)
                cv2.putText(frame, f"Distance {Distance} Inches",
                            (face_x - 6, face_y - 6), fonts, 0.5, BLACK, 2)


        # Navigation and edge detection
        img_edgerep = frame.copy()
        blur = cv2.bilateralFilter(img_edgerep, 9, 40, 40)
        edges = cv2.Canny(blur, 50, 150)
        img_edgerep_h = img_edgerep.shape[0] - 1
        img_edgerep_w = img_edgerep.shape[1] - 1

        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_edgerep, contours, -1, (0, 255, 0), 2)

        histogram = np.sum(edges[edges.shape[0] // 2:, :], axis=0)
        chunk = make_chunks(histogram, len(histogram) // StepSize)

        if len(chunk) > 0:
            chunk_pos = chunk.index(max(chunk))
            x_pos = chunk_pos * StepSize
            cv2.putText(frame, f"Chunk Position: {x_pos}", (10, 40), font, 1, GREEN, 2)

        if FC_X < int(frame.shape[1] / 2) - StepSize:
            direction = "Turn Left"
        elif FC_X > int(frame.shape[1] / 2) + StepSize:
            direction = "Turn Right"
        else:
            direction = "q Forward"

        cv2.putText(frame, direction, (10, img_edgerep_h - 20), font, 1, GREEN, 2)

        cv2.imshow("Frame", frame)
        cv2.imshow("Edges", img_edgerep)

        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
