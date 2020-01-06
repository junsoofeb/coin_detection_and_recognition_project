import cv2 as cv
import numpy as np
import time
try:
    import Image
except ImportError:
    from PIL import Image
import pytesseract
'''
# 한글 
print(pytesseract.image_to_string(Image.open('hangul.png'), lang='Hangul'))
'''

def make_hsv(image):
    blur = cv.GaussianBlur(image, (3,3), 0)
    hsv = cv.cvtColor(blur, cv.COLOR_RGB2HSV)

    lower_color = np.array([108, 23, 82])
    upper_color = np.array([179, 255, 255])

    '''
    [0, 58, 50] lower bound skin HSV
    [30, 255, 255] upper bound skin HSV
    '''
    mask = cv.inRange(hsv, lower_color, upper_color)
    blur = cv.medianBlur(mask, 5)

    #kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (8, 8)) 너무 손 두껍게 나와서 3으로 줄임
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    hsv = cv.dilate(blur, kernel)

    return hsv

def show(image, str = ""):
    cv.imshow(str, image)
    cv.waitKey()
    cv.destroyAllWindows()

def find_coin_roi(img):
    canny = cv.Canny(img, 50, 150)
    blur = cv.bilateralFilter(canny, 3, 30, 30)
    
    kernel = np.ones((4, 1), np.uint8)
    result = cv.erode(blur, kernel, iterations = 2)
    #show(result)
    
    kernel = np.ones((10, 33), np.uint8)
    result = cv.dilate(result, kernel, iterations = 1)
    #show(result)
    
    
    cnts = cv.findContours(result, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]
    
    max_index = None
    max_area = cv.contourArea(cnts[0])
    for i in range(len(cnts)):
        cnt = cnts[i]
        area = cv.contourArea(cnt)
        if max_area < area:
            max_area = area
            max_index = i    
    
    rect = cv.minAreaRect(cnts[max_index])    
    box = cv.boxPoints(rect)
    box = np.int0(box)
    
    alpha = 7
    box[0][1] = box[0][1] + alpha
    box[1][1] = box[1][1] - alpha
    box[2][1] = box[2][1] - alpha
    box[3][1] = box[3][1] + alpha
    
    x_set = set()
    y_set = set()
    
    for i in range(4):
        x_set.add(box[i][0])
        y_set.add(box[i][1])
    
    x_list = list(x_set)
    y_list = list(y_set)
    
    x_list = sorted(x_list)
    y_list = sorted(y_list)
    
    roi = img[y_list[0]:y_list[1], x_list[0]:x_list[1]]
    #show(roi)
    return roi


    
def distance_2_pixel(points):
    # points는 [x, y, radius]로 구성!
    x_list = []
    y_list = []
    
    for x, y, _ in points:
        x_list.append(x)    
        y_list.append(y)  
    
def biggest_circle(circles):
    r_list = []
    for _, _, r in circles:
        r_list.append(r)
        
    return max(r_list)    

def img_to_text(image):
    #t = str(int(time.time()))
    #cv.imwrite(f'roi_{t}.jpg', image)
    image = find_coin_roi(image)
    show(image, "coin_roi")

    image = cv.Canny(image, 70, 150)
    blur = cv.bilateralFilter(image, 5, 30, 50)
    #show(blur)
    try:
        text = pytesseract.image_to_string(blur, config='--psm 10 -c preserve_interword_spaces=1')
        print("OCR:",text)
    except:
        print("ocr err")
    
def grab_coin(image):
    # 동전 찾으면 1 반환, 못 찾으면 0 반환
    try:
        # hsv 색공간을 사용해 손과 배경 구별
        hsv_img = make_hsv(image.copy())
        #show(hsv_img)
        
        # hsv_img에 Opening을 적용해여 노이즈(하얀 점) 제거         
        kernel = np.ones((10, 10), np.uint8)
        hsv_img = cv.morphologyEx(hsv_img, cv.MORPH_OPEN, kernel)
        #show(hsv_img)
        
        dimension = image.shape
        w = dimension[1] # 640
        h = dimension[0] # 480
        
        # 첫번째 흰색점(손 끝) 찾기
        white_point = tuple()
        check = 0
        for x in range(w):
            for y in range(h):
                pixel_value = hsv_img[y, x]
                if pixel_value == 255:
                    white_point = (x, y)
                    check = 1
                    break
            if check == 1:
                break
                
        #print(white_point)
        #cv.circle(image, white_point, 10, (255, 0, 0), 3)
        #show(image)        
        
        
        # 조건 : 손 위치 (white_point)에서 일정 범위 안에 원의 중심이 있을 것!
        # 조건 박스 생성 및 범위 넘어갔을 경우 처리
        
        right = white_point[0] + 50
        left = white_point[0] - 100
        top = white_point[1] - 75
        bottom = white_point[1] + 75        
            
        roi = image.copy()[top:bottom, left:right]
        # roi에 허프 원 검출 알고리즘 사용
        gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
        #cv.HoughCircles(img, cv.HOUGH_GRADIENT, 해상도 비율(1주면 img와 동일), 원의 중심 사이간 거리, 등등)
        circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 15, param1 = 200, param2 = 30, minRadius = 2, maxRadius = 100)[0]
        # circles 는 [x, y, radius]로 구성!

        max_radius = biggest_circle(circles)

        # roi 박스 생성
        cv.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 4)

        # 이미지에 원 표시
        for c in circles:
            x = int(c[0] + left)
            y = int(c[1] + top)
            #print(x, y)
            try:
                if max_radius == c[2]:
                    cv.circle(image, (x, y), c[2], (0, 255, 0), 2)
                    # OCR 결과 포함
                    #img_to_text(roi)
                    show(image)
        
                    return 1
            except:
                pass
            
        return 0
        
    except:
        print("grap_coin_except!")
        return 0
        
    
def coin_on_hand(image):
    try:
        hsv = make_hsv(image)
        canny = cv.Canny(hsv, 50, 150)
        circles = cv.HoughCircles(canny, cv.HOUGH_GRADIENT, 1, 15, param1 = 200, param2 = 30, minRadius = 2, maxRadius = 100)[0]
        show(canny)
        for c in circles:
            cv.circle(image, (c[0], c[1]), c[2], (0, 255, 0), 2)
        show(image)
    except :
        print("coin_on_hand_except!")
    '''
    hsv = make_hsv(image)
    _, thre = cv.threshold(hsv, 127, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(thre, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(image, contours, -1, (0, 255, 0), 2)
    show(image)
    '''
def main():
    target = None
    cap = cv.VideoCapture(0)

    # 's' 키를 눌러서 target 이미지를 설정
    while(True):
        ret, frame = cap.read()
        hsv = make_hsv(frame.copy())
        cv.imshow('frame', frame)
        cv.imshow('hsv', hsv)
        key = cv.waitKey(30)
        if key == ord('s'):
            target = frame
            break
    cap.release()
    cv.destroyAllWindows()

    # 첫 번째 case 손으로 동전 집고 있는 경우
    result = grab_coin(target.copy())
    if result == 0:
        # 두 번째 case 손바닥에 동전 올려놓은 경우 
        #coin_on_hand(target.copy())
        pass
    else:
        print("find coin!")
        
    
    
while True:
    main()

