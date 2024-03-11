import picamera
import numpy as np
import imutils
import cv2
import time
from bookworm_module import BookWorm  # Replace 'bookworm_module' with the actual module name

# Define the lower and upper boundaries of the
# green circle in the HSV color space
colorLower = (29, 70, 6)
colorUpper = (75, 255, 255)

# Initialize Picamera
with picamera.PiCamera() as camera:
    camera.resolution = (640, 480)
    camera.framerate = 25

    # Initialize BookWorm
    bookworm = BookWorm()

    # Allow the camera to warm up
    time.sleep(0.1)

    # Keep looping
    for frame in camera.capture_continuous(format="bgr", use_video_port=True):
        # Grab the current frame
        image = frame.array

        # Blur the frame and convert to the HSV color space
        blurred = cv2.GaussianBlur(image, (11, 11), 0)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Construct a mask for the color "green", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        mask = cv2.inRange(hsv, colorLower, colorUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # Find counters in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
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
                # Then update the list of tracked points
                cv2.circle(image, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(image, center, 2, (0, 0, 255), -1)

                # Process the captured image with BookWorm
                bookworm.process_image(image)

        # Show the frame to the screen
        cv2.imshow("Frame", image)
        key = cv2.waitKey(1) & 0xFF

        # Press the 'q' key to stop the video stream
        if key == ord("q"):
            break

    # Clean up resources
    bookworm.cleanup()
