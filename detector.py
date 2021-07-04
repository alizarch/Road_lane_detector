import matplotlib.pylab as plt
import cv2
import numpy as np
from helper import region_of_interest , drow_the_lines ,process

############ 1 ############
def road_lane_detect_from_image():
    image = cv2.imread('road_img_2.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(image.shape)
    height = image.shape[0]
    width = image.shape[1]
    region_of_interest_vertices = [
        (0, height),
        (width/2.1, height/1.9),
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
        rho=6,
        theta=np.pi/180,
        threshold=160,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=25
    )
    image_with_lines = drow_the_lines(image, lines)
    plt.imshow(image_with_lines)
    plt.show()
        
############ 2 ############        
def road_lane_detect_from_video():
    cap = cv2.VideoCapture('road_vid_2.mp4')
    while cap.isOpened():
        ret, frame = cap.read()
        frame = process(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

############ 3 ############
def main():
    print("Enter 1, detact lane from image")
    print("Enter 2, detact lane from video")
    choice = input("Choose a right number")
    
    if choice == "1":
        road_lane_detect_from_image()
    elif choice == "2":
        road_lane_detect_from_video()
    else:
        print("Please choose a right number")

if __name__ == "__main__":
    main()
