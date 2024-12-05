import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import tensorflow as tf
from datetime import datetime

class ProductDetector:
    def __init__(self):
        # Product catalog with prices
        self.product_catalog = {
            'dove shampoo': 185,
            'lays': 10,
            'marble cake': 30,
            'maaza': 42,
            'munch': 5,
            'thums up': 50,
            'timepass biscuit': 25
        }
        
        # YOLO object detection model
        self.detection_model = YOLO('yolov8n.pt')
        
        # Product classification model
        self.classification_model = self._load_product_classifier()
        
        # Tracking variables
        self.detected_items = []
        self.total_bill = 0
        
    def _load_product_classifier(self):
        # Your existing model loading code
        products_model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, kernel_size=(3, 3), padding='valid', activation='relu', input_shape=(224, 224, 3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'),
            
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'),
            
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'),
            
            tf.keras.layers.Flatten(),
            
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(7, activation='softmax')
        ])
        products_model.load_weights('custom_model.h5')
        return products_model
    
    def _preprocess_image(self, image):
        target_size = (224, 224)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(image_rgb, target_size)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    
    def _make_square_padded_image(self, image):
        h, w, _ = image.shape
        max_side = max(h, w)
        
        square_img = np.ones((max_side, max_side, 3), dtype=np.uint8) * 255
        
        x_center = (max_side - w) // 2
        y_center = (max_side - h) // 2
        
        square_img[y_center:y_center+h, x_center:x_center+w] = image
        
        return square_img
    
    def _draw_invoice_panel(self, img, detected_items, total_bill):
        height, width, _ = img.shape
        panel_width = min(400, width // 3)  # Set the panel width to 1/3 of the screen width (or max 400)

        # Extend the gray space to the entire window
        white_panel = np.ones((height, panel_width, 3), dtype=np.uint8) * 255

        # Invoice Header
        cv2.putText(white_panel, "SMART INVOICE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 128), 2, cv2.LINE_AA)
        cv2.line(white_panel, (20, 60), (panel_width - 20, 60), (0, 0, 128), 2)  # Underline

        # Date and Time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(white_panel, current_time, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        # Item List Header
        cv2.putText(white_panel, "Items:", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.line(white_panel, (20, 150), (panel_width - 20, 150), (0, 0, 0), 1)  # Separator Line

        # Item List
        for i, (item, price) in enumerate(detected_items, 1):
            item_text = f"{i}. {item.capitalize()}"
            price_text = f"Rs.{price}"
            
            cv2.putText(white_panel, item_text, (20, 160 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(white_panel, price_text, (panel_width - 100, 160 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

        # Total
        cv2.line(white_panel, (20, height - 100), (panel_width - 20, height - 100), (0, 0, 0), 2)
        total_text = f"TOTAL: Rs.{total_bill}"
        cv2.putText(white_panel, total_text, (20, height - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 128), 2, cv2.LINE_AA)

        # Combine the panel with the original image and ensure the panel is fully visible
        img_with_invoice = np.hstack((img, white_panel))
        combined_width = img_with_invoice.shape[1]

        # Resize the window to fit the combined image width
        cv2.namedWindow("Smart Invoice Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Smart Invoice Detection", combined_width, height)

        return img_with_invoice

    def detect_and_track(self):
        cap = cv2.VideoCapture(0)
        cap.set(3, 1280)
        cap.set(4, 720)

        prediction_counts = {item: 0 for item in self.product_catalog}

        while True:
            success, img = cap.read()
            if not success:
                break

            results = self.detection_model(img, stream=True)

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_name = r.names[int(box.cls[0])]

                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    if class_name != "person":
                        cropped_img = img[y1:y2, x1:x2]
                        padded_img = self._make_square_padded_image(cropped_img)
                        processed_img = self._preprocess_image(padded_img)

                        prediction = self.classification_model.predict(processed_img)

                        if np.max(prediction) > 0.9:
                            predicted_class_index = np.argmax(prediction)
                            predicted_class = list(self.product_catalog.keys())[predicted_class_index]

                            prediction_counts[predicted_class] += 1

                            if prediction_counts[predicted_class] > 2 and predicted_class not in [item[0] for item in self.detected_items]:
                                self.detected_items.append((predicted_class, self.product_catalog[predicted_class]))
                                self.total_bill += self.product_catalog[predicted_class]

                            cvzone.putTextRect(img, predicted_class, (max(0, x1), max(40, y1)))
                    else:
                        cvzone.putTextRect(img, class_name, (max(0, x1), max(40, y1)))

            img_with_invoice = self._draw_invoice_panel(img, self.detected_items, self.total_bill)

            cv2.imshow("Smart Invoice Detection", img_with_invoice)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.total_bill = 0
                self.detected_items.clear()

        cap.release()
        cv2.destroyAllWindows()


# Run the detection
detector = ProductDetector()
detector.detect_and_track()