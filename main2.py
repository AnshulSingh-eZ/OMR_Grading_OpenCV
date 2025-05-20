import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform

path = r"D:\SR_Works\2nd_Task\Q7\omr.jpeg"
image = cv2.imread(path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gblur = cv2.GaussianBlur(gray, (5, 5), 0)
canny = cv2.Canny(gblur, 75, 200)


cnts = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
paperCnt = None

if len(cnts) > 0:
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            paperCnt = approx
            break

if paperCnt is not None:
    warped_gray = four_point_transform(gray, paperCnt.reshape(4, 2))
    image = four_point_transform(image, paperCnt.reshape(4, 2))
    gblur = cv2.GaussianBlur(warped_gray, (5, 5), 0)
    thresh = cv2.threshold(gblur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
else:
    thresh = cv2.threshold(gblur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

circles = cv2.HoughCircles(gblur, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=10, maxRadius=20)
ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}
correct = 0
detected_bubbles = []

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        detected_bubbles.append((i[0], i[1], i[2]))
    detected_bubbles = sorted(detected_bubbles, key=lambda x: x[1])
    question_groups = [detected_bubbles[i:i + 5] for i in range(0, len(detected_bubbles), 5)]
    for q, group in enumerate(question_groups):
        group = sorted(group, key=lambda x: x[0])
        bubbled = None
        for j, (x, y, r) in enumerate(group):
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.circle(mask, (x, y), r, 255, -1)
            masked = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(masked)
            if bubbled is None or total > bubbled[0]:
                bubbled = (total, j)
        color = (0, 0, 255)
        if q not in ANSWER_KEY:
            continue
        k = ANSWER_KEY[q]
        if k == bubbled[1]:
            color = (0, 255, 0)
            correct += 1
        cv2.circle(image, (group[k][0], group[k][1]), group[k][2], color, 2)

score = float(correct / 5.0) * 100
print("[INFO] Score: {:.2f}%".format(score))
cv2.putText(image, "{:.2f}%".format(score), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
cv2.imshow("Exam", image)
cv2.waitKey(0)
cv2.destroyAllWindows()