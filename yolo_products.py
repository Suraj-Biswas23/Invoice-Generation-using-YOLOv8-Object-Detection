import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import tensorflow as tf
from datetime import datetime

class ProductDetector:
    """
    A class that handles the product detection and tracking in real-time using YOLO for object detection 
    and a custom classification model to identify products and calculate the total bill.
    """
    
    def __init__(self):
        """
        Initializes the ProductDetector object with the product catalog, YOLO model, product classifier model, 
        and tracking variables for detected items and total bill.
        """
        self.product_catalog = {
            'dove shampoo': 185,
            'lays': 10,
            'marble cake': 30,
            'maaza': 42,
            'munch': 5,
            'thums up': 50,
            'timepass biscuit': 25
        }
        
        # Load YOLO object detection model
        self.detection_model = YOLO('yolov8n.pt')
        
        # Load product classification model
        self.classification_model = self._load_product_classifier()
        
        # Tracking variables
        self.detected_items = []
        self.total_bill = 0

    def _load_product_classifier(self):
        """
        Loads the pre-trained product classification model.
        
        Returns:
            tf.keras.Model: The product classification model.
        """
        # Load the entire model (architecture + weights)
        products_model = tf.keras.models.load_model('full_model.h5')
        
        return products_model

    def _preprocess_image(self, image):
        """
        Preprocesses the input image for prediction by resizing and normalizing it.

        Args:
            image (numpy.ndarray): The input image to preprocess.

        Returns:
            numpy.ndarray: The preprocessed image ready for classification.
        """
        target_size = (224, 224)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(image_rgb, target_size)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    
    def _make_square_padded_image(self, image):
        """
        Pads the input image to make it square while maintaining the aspect ratio.

        Args:
            image (numpy.ndarray): The input image to pad.

        Returns:
            numpy.ndarray: The square padded image.
        """
        h, w, _ = image.shape
        max_side = max(h, w)
        
        square_img = np.ones((max_side, max_side, 3), dtype=np.uint8) * 255
        
        x_center = (max_side - w) // 2
        y_center = (max_side - h) // 2
        
        square_img[y_center:y_center+h, x_center:x_center+w] = image
        
        return square_img
    
    def _draw_invoice_panel(self, img, detected_items, total_bill):
        """
        Draws the invoice panel on the image displaying the detected items and total bill.

        Args:
            img (numpy.ndarray): The input image on which the invoice panel will be drawn.
            detected_items (list): A list of tuples containing detected items and their prices.
            total_bill (float): The total amount for the detected items.

        Returns:
            numpy.ndarray: The image with the invoice panel.
        """
        height, width, _ = img.shape
        panel_width = min(400, width // 3)

        # Create white panel for invoice
        white_panel = np.ones((height, panel_width, 3), dtype=np.uint8) * 255

        # Add invoice details (header, date, items, total)
        cv2.putText(white_panel, "SMART INVOICE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 128), 2, cv2.LINE_AA)
        cv2.line(white_panel, (20, 60), (panel_width - 20, 60), (0, 0, 128), 2)

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(white_panel, current_time, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.putText(white_panel, "Items:", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.line(white_panel, (20, 150), (panel_width - 20, 150), (0, 0, 0), 1)

        for i, (item, price) in enumerate(detected_items, 1):
            item_text = f"{i}. {item.capitalize()}"
            price_text = f"Rs.{price}"
            
            cv2.putText(white_panel, item_text, (20, 160 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(white_panel, price_text, (panel_width - 100, 160 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.line(white_panel, (20, height - 100), (panel_width - 20, height - 100), (0, 0, 0), 2)
        total_text = f"TOTAL: Rs.{total_bill}"
        cv2.putText(white_panel, total_text, (20, height - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 128), 2, cv2.LINE_AA)

        img_with_invoice = np.hstack((img, white_panel))
        combined_width = img_with_invoice.shape[1]

        cv2.namedWindow("Smart Invoice Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Smart Invoice Detection", combined_width, height)

        return img_with_invoice

    def detect_and_track(self):
        """
        Detects products in real-time using the webcam and updates the total bill and detected items.
        Displays the result with a real-time invoice panel showing detected items and the total.
        """
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

                        if np.max(prediction) > 0.95:
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