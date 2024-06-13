
python tools/analysis_tools/analyze_logs.py plot_curve \
    work_dirs/resnet50_resnet18_mobilenetv2_bs256/20240613_151730/vis_data/scalars.json work_dirs/resnet50_bs256/20240613_144919/vis_data/scalars.json \
    --keys accuracy/top1 \
    --legend student teacher \
    --out ./plot/acc.png \
    --title Accuracy/Top1