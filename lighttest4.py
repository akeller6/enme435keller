from picamera2 import Picamera2
import numpy as np
import imutils
import cv2
import time

# Define the lower and upper boundaries of the
# green circle in the HSV color space
colorLower = (29, 70, 6)
colorUpper = (75, 255, 255)

# Initialize Picamera2
with Picamera2() as camera:
    camera.resolution = (640, 480)
    camera.framerate = 25

    # Allow the camera to warm up
    time.sleep(0.1)

    # Create an array to store the image
    image = np.empty((480, 640, 3), dtype=np.uint8)

    # Keep looping
    while True:
        # Capture a frame into the array
        camera.capture(image, format="bgr", use_video_port=True)

        # Blur the frame and convert to the HSV color space
        blurred = cv2.GaussianBlur(image, (11, 11), 0)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Construct a mask for the color "green", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        mask = cv2.inRange(hsv, colorLower, colorUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # Find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None

        # Proceed regardless to keep video streaming
        if len(cnts) > 0:
            # Find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            if radius > 0:
                # Draw the circle and centroid on the frame
                cv2.circle(image, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(image, center, 2, (0, 0, 255), -1)

        # Show the frame to the screen
        cv2.imshow("Frame", image)
        key = cv2.waitKey(1) & 0xFF

        # Press the 'q' key to stop the video stream
        if key == ord("q"):
            break

    # Clean up resources
    cv2.destroyAllWindows()
