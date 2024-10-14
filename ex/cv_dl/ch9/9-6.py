from pixellib.semantic import semantic_segmentation
import cv2 as cv

cap=cv.VideoCapture(0)

seg_video=semantic_segmentation()
seg_video.load_ade20k_model('deeplabv3_xception65_ade20k.h5')

seg_video.process_camera_ade20k(cap,overlay=True,frames_per_second=2,output_video_name='output_video.mp4',show_frames=True,frame_name='Pixellib')

cap.release()
cv.destroyAllWindows()