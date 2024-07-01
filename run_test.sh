GPU_NUM=0

for i in 1004
do
python test_all.py \
--dataset_type 'nuscenes' \
--model_name 'Scratch' \
--target 'vehicle' \
--exp_id $i \
--gpu_num $GPU_NUM \
--visualization 0
done

