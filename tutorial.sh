# Inference 한 줄 실행 (Inference 폴더에서 순서대로 실행)
# 1) Recon
# python recon/test.py --dataroot ../datasets/BCI_HER2 --name reconstruction --checkpoints_dir ckpt --results_dir results/A/recon --model cpt --CUT_mode CUT --gpu_ids 0 --netD n_layers --ndf 32 --netG resnet_9blocks --n_layers_D 5 --normG instance --normD instance --weight_norm spectral --lambda_GAN 1.0 --lambda_NCE 10.0 --nce_layers 0,4,8,12,16 --nce_T 0.07 --num_patches 256 --lambda_style 100.0 --lambda_content 1.0 --lambda_gp 10.0 --gp_weights '[0.015625,0.03125,0.0625,0.125,0.25,1.0]' --lambda_asp 10.0 --asp_loss_mode lambda_linear --use_simsiam True --use_clsA 0 --use_clsB 1 --use_clsfB 1 --lambda_cls 10.0 --cls_content False --lambda_discls 0.1 --dataset_mode aligned --direction AtoB --num_threads 2 --batch_size 1 --load_size 1024 --crop_size 1024 --preprocess crop --flip_equivariance False --display_winsize 512 --phase val --num_test 10000 --epoch latest
# 2) recon_dataset 준비
# python prepare_recon_dataset.py A
# 3) Classification
# python classification/test.py --ckpt_dir ckpt/classification --output_dir results/A/classification --data_root results/A/recon_dataset --mode AB --is_pred --fold all
