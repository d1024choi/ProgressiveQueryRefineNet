GPU_NUM=0
HOST_NODE_ADDR=10000
W_VEH=1.0
W_PED=0.0
W_ROAD=1e-2
W_LANE=0.0
W_INTM=1.0
W_OFF=0.0
BS=1
NUM_EPOCH=28
LR=2e-4
LR_SCHEDULE=1
DDP=1
DECODER_TYPE='mask'
NUM_HIERACHY=3
ADD_HEADER=1
CROSS_ATTN_METHOD=0
LEARN_OFFSET=0
USE_VIS_OFFSET=0
FEAT_INTERACTION=2
FEAT_INT_REPEAT=5

for i in 1004
do
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 --max_restarts=3 --rdzv_id=$GPU_NUM --rdzv_backend=c10d --rdzv_endpoint=localhost:$HOST_NODE_ADDR train.py \
--dataset_type 'nuscenes' \
--exp_id $i \
--gpu_num $GPU_NUM \
--num_cores 16 \
--num_epochs $NUM_EPOCH \
--batch_size $BS \
--w_vehicle $W_VEH \
--w_pedestrian $W_PED \
--w_road $W_ROAD \
--w_lane $W_LANE \
--w_intm $W_INTM \
--w_offset $W_OFF \
--learning_rate $LR \
--apply_lr_scheduling $LR_SCHEDULE \
--ddp $DDP \
--feat_inter_method $FEAT_INTERACTION \
--feat_inter_repeat $FEAT_INT_REPEAT \
--hierarchy_depth $NUM_HIERACHY \
--decoder_type $DECODER_TYPE \
--bool_add_dec_header $ADD_HEADER \
--cross_attn_method $CROSS_ATTN_METHOD \
--bool_learn_offset $LEARN_OFFSET \
--bool_use_vis_offset $USE_VIS_OFFSET \
--bool_apply_crosshead 0 \
--img_aug_prob 0.0 \
--max_translation 0
done

