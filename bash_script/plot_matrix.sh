
# # Stage1
# python tools/analysis_tools/confusion_matrix.py \
#     configs/cifar10_stage1/resnet50_bs256.py \
#     work_dirs/resnet50_bs256/epoch_100.pth \
#     --show \
#     --show-path ./plot/stage1/resnet50_teacher.png \
#     --include-values

# Stage2
python tools/analysis_tools/confusion_matrix.py \
    configs/cifar10_stage2/resnet50_resnet18_mobilenetv2_bs256.py \
    work_dirs/resnet50_resnet18_mobilenetv2_bs256/epoch_100.pth \
    --show \
    --show-path ./plot/stage2/resnet50_resnet18_mobilenet_student.png \
    --include-values
