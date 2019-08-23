# train nearest for x4
CUDA_VISIBLE_DEVICES=2 python main.py --dir_data /home/datasets/sr --data_train ImageNet3K --data_test Set5 --lr_downsample nearest --scale 4
# train DN for x4
CUDA_VISIBLE_DEVICES=2 python main.py --dir_data /home/datasets/sr --data_train ImageNet3K --data_test Set5 --lr_downsample DN --scale 4
# train blur for x4
CUDA_VISIBLE_DEVICES=2 python main.py --dir_data /home/datasets/sr --data_train ImageNet3K --data_test Set5 --lr_downsample blur --scale 4

# # test DN 
python main.py --dir_data /home/datasets/sr --data_test Urban100 --lr_downsample DN --test_only --save_results

# # test nearest 
python main.py --dir_data /home/datasets/sr --data_test Set5 --lr_downsample nearest --test_only --save_results