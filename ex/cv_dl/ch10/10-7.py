import cv2 as cv
import mediapipe as mp

mp_mesh=mp.solutions.face_mesh
mp_drawing=mp.solutions.drawing_utils
mp_styles=mp.solutions.drawing_styles

mesh=mp_mesh.FaceMesh(max_num_faces=2,refine_landmarks=True,min_detection_confidence=0.5,min_tracking_confidence=0.5)

cap=cv.VideoCapture(0,cv.CAP_DSHOW)

while True:
    ret,frame=cap.read()
    if not ret:
      print('프레임 획득에 실패하여 루프를 나갑니다.')
      break
    
    res=mesh.process(cv.cvtColor(frame,cv.COLOR_BGR2RGB))
    
    if res.multi_face_landmarks:
        for landmarks in res.multi_face_landmarks:
            mp_drawing.draw_landmarks(image=frame,landmark_list=landmarks,connections=mp_mesh.FACEMESH_TESSELATION,landmark_drawing_spec=None,connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(image=frame,landmark_list=landmarks,connections=mp_mesh.FACEMESH_CONTOURS,landmark_drawing_spec=None,connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(image=frame,landmark_list=landmarks,connections=mp_mesh.FACEMESH_IRISES,landmark_drawing_spec=None,connection_drawing_spec=mp_styles.get_default_face_mesh_iris_connections_style())
        
    cv.imshow('MediaPipe Face Mesh',cv.flip(frame,1))		# 좌우반전
    if cv.waitKey(5)==ord('q'):
      break

cap.release()
cv.destroyAllWindows()