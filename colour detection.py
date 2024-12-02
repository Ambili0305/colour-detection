import cv2
import numpy as np

# Function to get the HSV range for different colors
def get_hsv_range(color_name):
    if color_name == 'red':
        lower_hsv = np.array([0, 120, 70])
        upper_hsv = np.array([10, 255, 255])
        lower_hsv2 = np.array([170, 120, 70])
        upper_hsv2 = np.array([180, 255, 255])
        return [(lower_hsv, upper_hsv), (lower_hsv2, upper_hsv2)]
    
    elif color_name == 'blue':
        lower_hsv = np.array([100, 150, 0])
        upper_hsv = np.array([140, 255, 255])
        return [(lower_hsv, upper_hsv)]
    
    elif color_name == 'green':
        lower_hsv = np.array([40, 40, 40])
        upper_hsv = np.array([80, 255, 255])
        return [(lower_hsv, upper_hsv)]
    
    elif color_name == 'yellow':
        lower_hsv = np.array([20, 100, 100])
        upper_hsv = np.array([40, 255, 255])
        return [(lower_hsv, upper_hsv)]
    
    else:
        print("Color not supported. Using red as default.")
        return get_hsv_range('red')

# User input for the color selection
color_name = input("Enter the color to detect (red, blue, green, yellow): ").strip().lower()

# Load the image
image = cv2.imread(r'C:\Users\user\Desktop\red.jpg')  # Make sure the image path is correct

if image is None:
    print("Error: Image not found or cannot be opened.")
else:
    # Resize the input image to a smaller size (optional)
    resized_image = cv2.resize(image, (500, 500))  # Resize the image for better display

    # Convert the image from BGR (OpenCV default) to HSV
    hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)

    # Get the HSV ranges for the selected color
    hsv_ranges = get_hsv_range(color_name)

    # Create the mask for the selected color
    mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
    
    for lower_hsv, upper_hsv in hsv_ranges:
        mask += cv2.inRange(hsv_image, lower_hsv, upper_hsv)

    # Apply the mask to the original image
    result = cv2.bitwise_and(resized_image, resized_image, mask=mask)

    # Resize the result image for display
    result_resized = cv2.resize(result, (500, 500))  # Resize the result for better display

    # Display the original image and the result side by side
    cv2.imshow('Original Image', resized_image)
    cv2.imshow(f'{color_name.capitalize()} Color Detection', result_resized)

    # Wait for a key press to close the windows
    cv2.waitKey(1)  # Ensures the window is updated and interactive
    cv2.waitKey(0)  # Wait indefinitely for any key press
    cv2.destroyAllWindows()

    # Optionally, save the output image
    cv2.imwrite(r'C:\Users\user\Desktop\color_detection_output.jpg', result_resized)

