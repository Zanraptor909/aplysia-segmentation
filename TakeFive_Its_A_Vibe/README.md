# training command
change any settings you want in train.py, then run below command

python C:\TakeFive_Its_A_Vibe\src\train.py

alpysia_seg is the 8 photo hand labeled
aplysia_seg2 is the  25 photo ai labeled no review
seg_v8s is the 25 photo ai labeled and reviewed



# validate
choose a python script to run and change any settings you want. Change the test image file path too.

predict_count.py is the most accurate using the yolov8 segmentation
predict_count.py is the same, but uses webcam feed with rolling average
predict_split_merge is a custom segmentation
predict_color_hr_progress.py is custom segmenation with color heuristics