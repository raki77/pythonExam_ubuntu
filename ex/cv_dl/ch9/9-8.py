from pixellib.instance import instance_segmentation
import cv2 as cv

cap=cv.VideoCapture(0)

seg_video=instance_segmentation()
seg_video.load_model("mask_rcnn_coco.h5")

target_class=seg_video.select_target_classes(person=True,book=True)
seg_video.process_camera(cap,segment_target_classes=target_class,frames_per_second=2,show_frames=True,frame_name='Pixellib',show_bboxes=True)

cap.release()
cv.destroyAllWindows()