# Invoice-Generation-using-YOLOv8-Object-Detection

This project utilizes object detection and classification models to identify products in real-time using a webcam. The system employs YOLOv8 for detecting items and a custom-trained neural network to classify products. Detected items are displayed along with their prices in a real-time invoice, allowing users to view and track their total bill. The application is designed to help automate the process of shopping or inventory management by recognizing and calculating the cost of products detected on camera.

# Problem Statement
In traditional retail environments, customers must manually pick items, proceed to a cashier or self-checkout, and handle payments—often leading to long wait times and inefficiencies. The aim is to automate this process in unmanned stores, enabling a seamless shopping experience where products are automatically added to an invoice on the app as customers interact with shelves, removing the need for shop personnel.

# Solution
Our solution is a Smart Invoice Detection System that leverages advanced computer vision techniques to track both the customer and the items they interact with. The system identifies products in real-time using object detection and classification, generating an invoice as items are picked. This eliminates the need for cashiers, allowing customers to pay directly and leave without manual intervention.

# Tech Stack
 - Programming Language: Python
 - Libraries and Frameworks:
   * OpenCV: For image processing and video feed management
   * YOLO (You Only Look Once): For real-time object detection
   * TensorFlow/Keras: For training and deploying the custom classification model
   * NumPy: For numerical computations
 - Model Files:
   * Custom Model Training Script: CV_Model_Train.py
   * Model Architecture and Weights:
     * full_model.h5
     * custom_model.weights.h5

# Approach
 * Object Detection: Utilized the YOLOv8 model to detect products and the customer in the live video feed.
 * Custom Classification Model:
   - Developed a custom classification model on top of YOLO to accurately classify detected products.
   - Training involved extensive data augmentation to handle variations in lighting, angles, and product orientations.
 * Training and Testing:
   - Created custom datasets for training and testing.
   - Fine-tuned model parameters, including learning rate, batch size, and architecture, to achieve high accuracy.
 * Real-Time Tracking and Invoice Generation:
   - Processed live video input to track the customer and their interactions with products.
   - Added detected products to a virtual invoice, updating the total bill in real-time.

## Screenshots

**1. Model Performance:**

   ![image](https://github.com/user-attachments/assets/0c9f09da-5faa-4090-86d0-37f13eb7bc19)

**2. UI Screen:**

   ![Readme Screenshot](https://github.com/user-attachments/assets/d4146982-c88c-475b-9056-220e0e8a8491)

## Usage

1. You can create your own dataset to train your own model or use the already trained custom model from the repo.
   
2. Clone the repository
```
git clone https://github.com/Suraj-Biswas23/Invoice-Generation-using-YOLOv8-Object-Detection.git
```

3. Install dependencies
```
pip install -r requirements.txt
```

4. Run the Python File
```
python invoice_products_yolo.py
```
<br/>

*Note: Install the dependencies in a virtual environment to prevent dependencies from clashing with other projects. You can also try using tensorflow-gpu for good performance by installing the dependency separately.*
