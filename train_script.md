# Monitor VRAM usage while training
while ($true) { clear; nvidia-smi; sleep 5 }

python Read_dataset/VideoDataset.py --root dataset/CAER/caer_3dmm_half/caer_3dmm_half --split train

python.exe video_train.py `
  --data_dir dataset/CAER/caer_3dmm `
  --val_split validation `
  --num_classes 7 `
  --backbone_checkpoint checkpoints/caers_MA3D.pth `
  --use_3dmm `
  --stats_path dataset/CAER/caer_3dmm/video_3dmm_stats.npz `
  --temporal_module transformer `
  --max_frames 16 `
  --frame_step 2 `
  --batch_size 32 `
  --lr 1e-4 `
  --weight_decay 5e-4 `
  --epochs 20 `
  --patience 5 `
  --unfreeze_backbone_epoch 5 `
  --backbone_lr_scale 0.1 `
  --use_class_weights `
  --cb_loss_pow 0.5 `
  --use_weighted_sampler `
  --cb_sampler_pow 0.3 `
  --select_metric mean `
  --resume_name "video_best_R1.pth" `
  --log_file "video_log_R1.txt"


python inference_video.py --checkpoint "checkpoints/video_best_R1.pth" --num_workers 0 --save_pdf "inference_report_R1.pdf"


train DFEW


python video_train.py \
  --data_type dfew \
  --data_dir Datasets \
  --val_split test \
  --num_classes 7 \
  --backbone_checkpoint checkpoints/dfew_MA3D.pth \
  --use_3dmm \
  --stats_path Datasets/DFEW/video_3dmm_stats.npz \
  --temporal_module transformer \
  --max_frames 16 \
  --frame_step 2 \
  --batch_size 32 \
  --lr 1e-4 \
  --weight_decay 5e-4 \
  --epochs 20 \
  --patience 5 \
  --unfreeze_backbone_epoch 5 \
  --backbone_lr_scale 0.1 \
  --use_class_weights \
  --cb_loss_pow 0.5 \
  --use_weighted_sampler \
  --cb_sampler_pow 0.3 \
  --select_metric mean \
  --resume_name "video_best_dfew_1.pth" \
  --log_file "video_log_dfew_2.txt"