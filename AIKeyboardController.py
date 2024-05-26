import cv2
from cvzone.HandTrackingModule import HandDetector
from time import sleep
from pynput.keyboard import Controller


cap = cv2.VideoCapture(0)

cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector()
finalText = ''

#all the keys are stored in a list
keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]]

keyboard= Controller

def drawAll(img, buttonList): #a function to actually draw buttons on the screen
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        cv2.rectangle(img, button.pos, (x + w, y + h), (150, 0, 0), cv2.FILLED)
        cv2.putText(img, button.text, (x + 20, y + 60), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
    return img


# def rectangle_area(p1, p2, p3, p4):
#     """Calculate the area of a rectangle given its four coordinates using NumPy."""
#     points = np.array([p1, p2, p3, p4])
#
#     # Calculate the distances between adjacent points
#     dists = np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1)
#
#     # Find the lengths of two adjacent sides
#     side1 = np.min(dists)
#     side2 = np.max(dists)
#
#     # Calculate the area of the rectangle
#     area = side1 * side2
#
#     return area


class Button(): #button class to pass the attributes to drawAll function
    def __init__(self, pos, text, size = [85,85]):
        self.pos = pos
        self.size = size
        self.text = text

buttonList = [] #creates a  list of all buttons
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        buttonList.append(Button([100*j+50, 100*i+50], key))

while True:
    success, img = cap.read()

    img1 = detector.findHands(img)
    img = drawAll(img, buttonList)

    if img1 and img1[0]:
        fingerData = img1[0]
        boxValues = fingerData[0]['bbox']  # will find coordinates of bounding boxes
        area = boxValues[2] * boxValues[3] # will find area of bounding box
        # print(area)
        lmList = fingerData[0]['lmList']
        landmark_8 = lmList[8]
        landmark_12 = lmList[12]
        # print(landmark_8)
        cv2.circle(img, (landmark_8[0], landmark_8[1]), 10, (0, 0, 255), cv2.FILLED)
        cv2.circle(img, (landmark_12[0], landmark_12[1]), 10, (0, 0, 255), cv2.FILLED)

        if 40000<area<60000:   #the below code will only work when hands are at a particular distance from camera
            cv2.putText(img, 'Hands Distance OK', (500,500), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
            for button in buttonList:
                x, y = button.pos
                w, h = button.size
                if x < landmark_8[0] < x + w and y < landmark_8[1] < y + h:
                    cv2.rectangle(img, button.pos, (x + w, y + h), (200, 0, 0), cv2.FILLED)
                    cv2.putText(img, button.text, (x + 20, y + 60), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                    l, _, _ = detector.findDistance((landmark_8[0], landmark_8[1]), (landmark_12[0], landmark_12[1]),
                                                    img)  # finds the distance between landmark 8 and 12
                    # print(l)

                    if l <35:
                        keyboard.press(button.text)
                        cv2.rectangle(img, button.pos, (x + w, y + h), (0, 200, 0), cv2.FILLED)
                        cv2.putText(img, button.text, (x + 20, y + 60), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                        finalText += button.text
                        sleep(0.15)

    cv2.rectangle(img, (50,350), (700,450), (255, 255, 255), cv2.FILLED)
    cv2.putText(img, finalText, (60, 425), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

    cv2.imshow('Video Output', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
