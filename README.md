# coin_detection_and_recognition_project
with OpenCV


## 1. 개요

카메라로 동전이 어디에 있는지 찾고,(coin_detection.py)     
동시에 얼마짜리 동전인지 인식(coin_recognition.py)하는 프로그램

## 2. 구현 환경

1) ubuntu 18.04
2) Python 3.6
3) OpenCV-python 4.2.0
4) Numpy 1.17
5) pytesseract 0.3.1

## 3. coin_detection.py 동작 과정

### 조건 1. 손으로 동전을 잡고 있음.
### 조건 2. 배경에는 피부색과 유사한 색이 없음. 

1) 원본 이미지에서 hsv 모델을 사용하여, 배경에서 손가락의 위치를 파악  
2) 손가락 위치에서 일정 범위(빨간 박스처리)를 roi로 설정  
3) roi에서 HOUGH 원 변환 알고리즘을 사용하여 원을 찾음  
4) roi안에 원이 여러 개인 경우, 반지름이 가장 큰 원을 선택  

## 4. coin_detection.py 동작 사진

![스크린샷, 2020-01-07 03-18-07](https://user-images.githubusercontent.com/46870741/71838653-b2ed2800-30fc-11ea-9222-167f3d35d12b.png)

![스크린샷, 2020-01-07 03-18-13](https://user-images.githubusercontent.com/46870741/71838654-b2ed2800-30fc-11ea-8239-1a205f49a619.png)

![스크린샷, 2020-01-07 03-19-41](https://user-images.githubusercontent.com/46870741/71838656-b2ed2800-30fc-11ea-98bd-034a59ac44e3.png)

![스크린샷, 2020-01-07 03-19-44](https://user-images.githubusercontent.com/46870741/71838657-b2ed2800-30fc-11ea-9a0c-7deb13282ad0.png)


