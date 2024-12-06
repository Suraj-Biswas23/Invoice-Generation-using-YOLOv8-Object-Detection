# Invoice-Generation-using-YOLOv8-Object-Detection

This project utilizes object detection and classification models to identify products in real-time using a webcam. The system employs YOLOv8 for detecting items and a custom-trained neural network to classify products. Detected items are displayed along with their prices in a real-time invoice, allowing users to view and track their total bill. The application is designed to help automate the process of shopping or inventory management by recognizing and calculating the cost of products detected on camera.

## Screenshots

1. Model Performance:
   ![image](https://github.com/user-attachments/assets/0c9f09da-5faa-4090-86d0-37f13eb7bc19)
 
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
