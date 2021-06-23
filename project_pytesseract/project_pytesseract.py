#!/usr/bin/etc python

import cv2
import numpy as np
import pytesseract
from PIL import Image


class Recognition:
    def ExtractNumber(self):
        Number = 'C:/Users/DHICC/PycharmProjects/project_pytesseract/testimg3.jpg  ' #번호판 사진

        # 이미지 파일 읽기
        # cv2.IMREAD_COLOR : 이미지 파일을 Color로 읽음
        img = cv2.imread(Number, cv2.IMREAD_COLOR)

        #이미지 파일 보기 1
        cv2.imshow('Original', img)
        cv2.waitKey(0) # keyboard입력 대기(0: 무한대기)
        cv2.destroyAllWindows()

        copy_img = img.copy() # 이미지 수정을 위해 copy 생성
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 색상 공간 변환(BGR2GRAY: 그레이스케일 이미지)
        cv2.imwrite('gray.jpg', img2) # 변환된 이미지 저장

        #이미지 파일 보기 2
        cv2.imshow('Original', img2)
        cv2.waitKey(0) # keyboard입력 대기(0: 무한대기)
        cv2.destroyAllWindows()

        # 가우시안 블러링
        blur = cv2.GaussianBlur(img2, (3, 3), 0)
        cv2.imwrite('blur.jpg', blur)

        #이미지 파일 보기 3
        cv2.imshow('Original', blur)
        cv2.waitKey(0) # keyboard입력 대기(0: 무한대기)
        cv2.destroyAllWindows()

        # 엣지 검출
        canny = cv2.Canny(blur, 100, 200)
        cv2.imwrite('canny.jpg', canny)

        #이미지 파일 보기 4
        cv2.imshow('Original', canny)
        cv2.waitKey(0) # keyboard입력 대기(0: 무한대기)
        cv2.destroyAllWindows()

        # cv2.findContours(이진화 이미지, 검색 방법, 근사화 방법)
        # 반환값으로 윤곽선, 계층 구조를 반환
        # (cv2.RETR_TREE : 모든 윤곽선을 검출하며, 계층 구조를 모두 형성)
        # (cv2.CHAIN_APPROX_SIMPLE : 윤곽점들 단순화 수평, 수직 및 대각선 요소를 압축하고 끝점만 남김)
        contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        box1 = []
        f_count = 0
        select = 0
        plate_width = 0

        # ??????????전체 이미지에서 Contour의 가로 세로 비율 값과 면적을 통해, 번호판 영역에 벗어난 걸로 추정되는 값들은 제외 시켜주었습니다????????????

        for i in range(len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt) # cv2.contourArea: 객체의 넓이
            x, y, w, h = cv2.boundingRect(cnt) # 도형을 감싸는 사각형 영역 추출
            rect_area = w * h  # area size
            aspect_ratio = float(w) / h  # ratio = width/height

            if (aspect_ratio >= 0.2) and (aspect_ratio <= 1.0) and (rect_area >= 100) and (rect_area <= 700):
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
                box1.append(cv2.boundingRect(cnt))

        # 버블 정렬
        # 카운트가 가장 높은 Contour 값 = 번호판의 시작점
        for i in range(len(box1)):
            for j in range(len(box1) - (i + 1)):
                if box1[j][0] > box1[j + 1][0]:
                    temp = box1[j]
                    box1[j] = box1[j + 1]
                    box1[j + 1] = temp

        # 직사각형 사이의 길이를 측정하는 번호판을 찾기 위해
        for m in range(len(box1)):
            count = 0
            for n in range(m + 1, (len(box1) - 1)):
                delta_x = abs(box1[n + 1][0] - box1[m][0])
                if delta_x > 150:
                    break
                delta_y = abs(box1[n + 1][1] - box1[m][1])
                if delta_x == 0:
                    delta_x = 1
                if delta_y == 0:
                    delta_y = 1
                gradient = float(delta_y) / float(delta_x)
                if gradient < 0.25:
                    count = count + 1
            # measure number plate size
            if count > f_count:
                select = m
                f_count = count;
                plate_width = delta_x
        cv2.imwrite('snake.jpg', img)

        # 번호판의 사이즈는 상수 값으로 Offset을 주어서 추출
        number_plate = copy_img[box1[select][1] - 10:box1[select][3] + box1[select][1] + 20, box1[select][0] - 10:140 + box1[select][0]]

        resize_plate = cv2.resize(number_plate, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC + cv2.INTER_LINEAR)
        plate_gray = cv2.cvtColor(resize_plate, cv2.COLOR_BGR2GRAY)

        # 이미지 이진화(흑/백)
        ret, th_plate = cv2.threshold(plate_gray, 150, 255, cv2.THRESH_BINARY)
        cv2.imwrite('plate_th.jpg', th_plate)

        #이미지 파일 보기 5
        cv2.imshow('Original', th_plate)
        cv2.waitKey(0) # keyboard입력 대기(0: 무한대기)
        cv2.destroyAllWindows()

        # 1로 가득찬 array 생성(커널 생성)
        kernel = np.ones((3, 3), np.uint8)

        # 이미지 침식(흐릿한 경계를 배경으로 만듦)
        er_plate = cv2.erode(th_plate, kernel, iterations=1)
        er_invplate = er_plate
        cv2.imwrite('er_plate.jpg', er_invplate)

        #이미지 파일 보기 6
        cv2.imshow('Original', er_invplate)
        cv2.waitKey(0) # keyboard입력 대기(0: 무한대기)
        cv2.destroyAllWindows()

        #Tesseract!
        result = pytesseract.image_to_string(Image.open('er_plate.jpg'), 'kora', '--psm 7 --oem 0')
        arr=result.split('\n')[0:1]
        result='\n'.join(arr)
        return (result.replace(" ", " ")) # replace: 문자열 치환?


recogtest = Recognition()
result = recogtest.ExtractNumber()
print(result)







