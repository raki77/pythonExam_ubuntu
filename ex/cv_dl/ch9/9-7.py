from pixellib.instance import instance_segmentation
import cv2 as cv

seg=instance_segmentation()
seg.load_model("mask_rcnn_coco.h5")

img_fname='busy_street.jpg'
info,img_segmented=seg.segmentImage(img_fname,show_bboxes=True)

cv.imshow('Image segmention overlayed',img_segmented)

cv.waitKey()
cv.destroyAllWindows()