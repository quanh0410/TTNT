import cv2
from ultralytics import YOLO
from util import read_license_plate

license_plate_detector = YOLO('./models/license_plate_detector.pt')

image_path = './test.jpg'
image = cv2.imread(image_path)

license_plates = license_plate_detector(image)[0]
for license_plate in license_plates.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = license_plate

    license_plate_crop = image[int(y1):int(y2), int(x1):int(x2), :]
    cv2.imshow('license', license_plate_crop)
    cv2.waitKey(0)

    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
    license_plate_crop_thresh = cv2.adaptiveThreshold(license_plate_crop_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                    cv2.THRESH_BINARY_INV, 13, 2)
    
    cv2.imshow('license', license_plate_crop_thresh)
    cv2.waitKey(0)

    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

    if license_plate_text is not None:
        print(license_plate_text)

