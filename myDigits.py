import os
import cv2
import numpy as np

drawing = False  
ix, iy = -1, -1  

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

# Create a blank canvas
canvas = np.zeros((400, 400, 1), dtype=np.uint8)

# Create a named window and set the mouse callback
cv2.namedWindow('Draw Digit in center')
cv2.setMouseCallback('Draw Digit in center', drawDigit)

i = 0
while True:
    # Display the canvas
    cv2.imshow('Draw Digit in center', canvas)
    
    # Press 'p' to predict the digit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Get the bounding box of the largest contour
            x, y, w, h = cv2.boundingRect(contours[0])
            digit_crop = canvas[y:y+h, x:x+w]
            padded_crop = cv2.copyMakeBorder(digit_crop, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
            resized_crop = cv2.resize(padded_crop, (28, 28))
            cv2.imwrite(f"/Users/salonimacbook/Desktop/coding projects/DigitRecognitionNN/digits/9/{i}.png", resized_crop)
        else: 
            cv2.imwrite(f"/Users/salonimacbook/Desktop/coding projects/DigitRecognitionNN/digits/9/{i}.png", canvas)
        i += 1
    if key == ord('c'):
        canvas = np.zeros((400, 400, 1), dtype=np.uint8)
    
    # Press 'q' to quit
    if key == ord('q'):
        break

cv2.destroyAllWindows()