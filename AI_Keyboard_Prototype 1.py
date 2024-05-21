import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)

cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector()

#all the keys are stored in a list
keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]]



def drawAll(img, buttonList): #a function to actually draw buttons on the screen
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        cv2.rectangle(img, button.pos, (x + w, y + h), (255, 0, 0), cv2.FILLED)
        cv2.putText(img, button.text, (x + 20, y + 60), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
    return img

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
        fingerData= img1[0]
        lmList = fingerData[0]['lmList']
        landmark_8 = lmList[8]
        landmark_12 = lmList[12]
        # print(landmark_8)
        cv2.circle(img, (landmark_8[0], landmark_8[1]), 10, (0, 255, 0), cv2.FILLED)
        cv2.circle(img, (landmark_12[0], landmark_12[1]), 10, (0, 255, 0), cv2.FILLED)

        for button in buttonList:
            x,y = button.pos
            w,h = button.size
            if x< landmark_8[0] < x+w and y<landmark_8[1]<y+h:
                cv2.rectangle(img, button.pos, (x + w, y + h), (0,255, 0), cv2.FILLED)
                # detector.findDistance(landmark_8)

    cv2.imshow('Video Output', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
