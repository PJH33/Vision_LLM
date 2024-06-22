import cv2

camera_indices = [0, 1]
camera = None
for index in camera_indices:
    cam = cv2.VideoCapture(index)
    if cam.isOpened():
        camera = cam
        print(f"웹캠이 인덱스 {index}에서 열렸습니다.")
        break

if not camera or not camera.isOpened():
    print("웹캠을 열 수 없습니다.")
else:
    # 웹캠에서 프레임을 읽어와서 화면에 표시
    while True:
        success, frame = camera.read()
        if not success:
            print("프레임을 읽을 수 없습니다.")
            break

        cv2.imshow('Webcam Test', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()
