import cv2
import numpy as np

############ 1 ############
#Take region for processing
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

############ 2 ############
#Drow the line on detective lane
def drow_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1, y1), (x2, y2),
                     (255, 0, 255), thickness=5)

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img

############ 3 ############
#Processing on video frames
def process(image):
    height = image.shape[0]
    width = image.shape[1]
    region_of_interest_vertices=[
        (100, height),
        (width/2, height/1.9),
        (width, height)
    ]

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny_image = cv2.Canny(gray_image, 100, 200)
    cropped_image = region_of_interest(
        canny_image,
        np.array(
            [region_of_interest_vertices],
            np.int32
        )
    )

    lines = cv2.HoughLinesP(
        cropped_image,
        rho=2,
        theta=np.pi/180,
        threshold=50,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=100
    )
    image_with_lines = drow_the_lines(image, lines)
    return image_with_lines
