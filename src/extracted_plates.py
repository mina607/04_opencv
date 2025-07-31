import cv2
import numpy as np

win_name = "scanning"
img = cv2.imread("../img/car1.jpg")
print("❌ 이미지를 불러올 수 없습니다. 경로를 확인하세요.")
exit()


rows, cols = img.shape[:2]
draw = img.copy()
pts_cnt = 0
pts = np.zeros((4,2), dtype=np.float32)
scan_count = 0

def onMouse(event, x, y, flags, param):  #마우스 이벤트 콜백 함수 구현 ---① 
    global  pts_cnt, scan_count                     # 마우스로 찍은 좌표의 갯수 저장
    if event == cv2.EVENT_LBUTTONDOWN:  
        cv2.circle(draw, (x,y), 10, (0,255,0), -1) # 좌표에 초록색 동그라미 표시
        cv2.imshow(win_name, draw)

        pts[pts_cnt] = [x,y]            # 마우스 좌표 저장
        pts_cnt+=1
        if pts_cnt == 4:                       # 좌표가 4개 수집됨 
            # 좌표 4개 중 상하좌우 찾기 ---② 
            sm = pts.sum(axis=1)                 # 4쌍의 좌표 각각 x+y 계산
            diff = np.diff(pts, axis = 1)       # 4쌍의 좌표 각각 x-y 계산

            topLeft = pts[np.argmin(sm)]         # x+y가 가장 값이 좌상단 좌표
            bottomRight = pts[np.argmax(sm)]     # x+y가 가장 큰 값이 우하단 좌표
            topRight = pts[np.argmin(diff)]     # x-y가 가장 작은 것이 우상단 좌표
            bottomLeft = pts[np.argmax(diff)]   # x-y가 가장 큰 값이 좌하단 좌표

            # 변환 전 4개 좌표 
            pts1 = np.float32([topLeft, topRight, bottomRight , bottomLeft])

            # 변환 후 영상에 사용할 서류의 폭과 높이 계산 ---③ 
            w1 = abs(bottomRight[0] - bottomLeft[0])    # 상단 좌우 좌표간의 거리
            w2 = abs(topRight[0] - topLeft[0])          # 하당 좌우 좌표간의 거리
            h1 = abs(topRight[1] - bottomRight[1])      # 우측 상하 좌표간의 거리
            h2 = abs(topLeft[1] - bottomLeft[1])        # 좌측 상하 좌표간의 거리
            width = int(max(w1, w2))                       # 두 좌우 거리간의 최대값이 서류의 폭
            height = int(max(h1, h2))                      # 두 상하 거리간의 최대값이 서류의 높이
            
            # 변환 후 4개 좌표
            pts2 = np.float32([[0,0], [width-1,0], 
                                [width-1,height-1], [0,height-1]])

            # 변환 행렬 계산 
            mtrx = cv2.getPerspectiveTransform(pts1, pts2)
            # 원근 변환 적용
            result = cv2.warpPerspective(img, mtrx, (width, height))
            cv2.imshow('scanned', result)

            # 1. 그레이스케일로 변환
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

            # 2. 블러 처리 (노이즈 제거)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)

            # 3. 에지 검출 (윤곽선)
            edges = cv2.Canny(blur, 30, 100)

            # 4. 결과 확인용 창 추가
            cv2.imshow('edges', edges)

            # 5. 윤곽선 이미지 저장
            edge_filename = f'scanned_{scan_count}_edges.jpg'
            cv2.imwrite(edge_filename, edges)
            print(f"✅ 윤곽선 저장 완료: {edge_filename}")

            result = cv2.warpPerspective(img, mtrx, (width, height))
            cv2.imshow('scanned', result)

            # 🔽 윤곽선 처리 추가
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 75, 200)
            cv2.imshow('gray', gray)
            cv2.imshow("edges_original", edges)


            cv2.imwrite("gray.jpg", gray)
            cv2.imwrite("edges_original.jpg", edges)
            print("✅ 초기 흑백/윤곽선 이미지 저장 완료")
            pts_cnt = 0
            draw[:] = img[:]

cv2.imshow(win_name, img)
cv2.setMouseCallback(win_name, onMouse)    # 마우스 콜백 함수를 GUI 윈도우에 등록 ---④
cv2.waitKey(0)
cv2.destroyAllWindows()