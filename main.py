import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from PIL import Image



myImages = []
myLabels = []

for i in np.arange(10):
    myDigits = f'/Users/salonimacbook/Desktop/coding projects/DigitRecognitionNN/digits/{i}'
    for filename in os.listdir(myDigits):
        img = cv2.imread(os.path.join(myDigits, filename), cv2.IMREAD_GRAYSCALE)
        img = np.invert(img)
        img = img.astype(np.float32) / 255.0

        label = i
        myImages.append(img)
        myLabels.append(label)

myImages = np.array(myImages)
myLabels = np.array(myLabels)
mnist = tf.keras.datasets.mnist

(xTrain, yTrain), (xTest, yTest) = mnist.load_data()

xTrain = xTrain.astype(np.float32) / 255.0
xTest = xTest.astype(np.float32) / 255.0

xTrain = xTrain.reshape(-1, 28,28,1)
xTest = xTest.reshape(-1,28,28,1)

myImages = myImages.reshape(-1, 28, 28, 1)

xTrain= np.concatenate((xTrain, myImages), axis=0)
yTrain = np.concatenate((yTrain, myLabels), axis=0)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()


model.fit(xTrain, yTrain, epochs = 10)


test_loss, test_acc = model.evaluate(xTest, yTest, verbose=2)
print(f"Test accuracy: {test_acc}")
model.save('handwritten.model')


model = tf.keras.models.load_model('handwritten.model')

drawing = False  
ix, iy = -1, -1  

# Mouse for drawing
def drawDigit(event, x, y, flags, param):
    global drawing, ix, iy, canvas
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(canvas, (x, y), 8, (255, 255, 255), -1)  
    elif event == cv2.EVENT_LBUTTONUP:
        cv2.circle(canvas, (x, y), 8, (255, 255, 255), -1)  # Draw the final circle

def preprocess_digit(canvas):
    # Find contours to crop 
    contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        digit_crop = canvas[y:y+h, x:x+w]

        # Add padding around the digit
        pad = 10  
        padded_crop = cv2.copyMakeBorder(digit_crop, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

        # Resize the cropped image to 28x28 and normalize
        img = cv2.resize(padded_crop, (28, 28), interpolation=cv2.INTER_AREA)
        img = np.invert(img)  
        img = img / 255.0  # Normalize pixel values to range [0, 1]
        img = img.reshape(1, 28, 28, 1) 
        return img
    return None

# blank canvas
canvas = np.zeros((400, 400, 1), dtype=np.uint8)

cv2.namedWindow('Draw Digit')
cv2.setMouseCallback('Draw Digit', drawDigit)

while True:
    cv2.imshow('Draw Digit', canvas)
    
    # Press 'p' to predict the digit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('p'):
        # Preprocess the canvas image for the model
        img = preprocess_digit(canvas)
        if img is not None:
            # Predict the digit
            prediction = model.predict(img)
            digit = np.argmax(prediction)
            print(f"Predicted Digit: {digit}")
            
            # Display the prediction on the canvas
            cv2.putText(canvas, f"Prediction: {digit}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            print("No digit detected. Please draw a clear digit.")
    # Press 'c' to clear the canvas
    if key == ord('c'):
        canvas = np.zeros((400, 400, 1), dtype=np.uint8)
    
    # Press 'q' to quit
    if key == ord('q'):
        break

cv2.destroyAllWindows()
