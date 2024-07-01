import argparse

parser = argparse.ArgumentParser()

# Exp Info
parser.add_argument('--model_name', type=str, default='')
parser.add_argument('--exp_id', type=int, default=300)
parser.add_argument('--gpu_num', type=int, default=0)
parser.add_argument('--load_pretrained', type=int, default=0)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--ddp', type=int, default=0, help='set to 1 if you want to use DDP training')
parser.add_argument('--bool_mixed_precision', type=int, default=0, help='we did not validate if this option works well or not')
parser.add_argument('--num_cores', type=int, default=4, help='the number of cores for data preparation')

# Dataset
parser.add_argument('--dataset_dir', type=str, default='')
parser.add_argument('--label_dir', type=str, default='')
parser.add_argument('--save_dir', type=str, default='')
parser.add_argument('--dataset_type', type=str, default='nuscenes')
parser.add_argument('--val_ratio', type=float, default=0.05, help='val_ratio x total_num_train_data will be used for validation process')

parser.add_argument('--img_h', type=int, default=224, help='target input image height to backbone net')
parser.add_argument('--img_w', type=int, default=480, help='target input image width to backbone net')
parser.add_argument('--img_top_crop', type=int, default=46)
parser.add_argument('--bev_h', type=int, default=200, help='height of BEV map')
parser.add_argument('--bev_w', type=int, default=200, help='width of BEV map')
parser.add_argument('--bev_h_meters', type=int, default=100)
parser.add_argument('--bev_w_meters', type=int, default=100)
parser.add_argument('--bev_offset', type=int, default=0)

parser.add_argument('--target', type=str, default='none', help='should be none if it is not used')

# Training Env
parser.add_argument('--num_epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=1e-7)
parser.add_argument('--min_learning_rate', type=float, default=0.00001)
parser.add_argument('--grad_clip', type=float, default=0.0)

parser.add_argument('--w_vehicle', type=float, default=0.0, help='loss weight for vehicle')
parser.add_argument('--w_pedestrian', type=float, default=0.0, help='loss weight for pedestrian')
parser.add_argument('--w_road', type=float, default=0.0, help='loss weight for road')
parser.add_argument('--w_lane', type=float, default=0.0, help='loss weight for lane')
parser.add_argument('--w_intm', type=float, default=0.0, help='loss weight for auxiliary task')
parser.add_argument('--w_offset', type=float, default=0.0, help='loss weight for auxiliary task')

parser.add_argument('--valid_step', type=int, default=1)
parser.add_argument('--save_every', type=int, default=3)
parser.add_argument('--max_num_chkpts', type=int, default=5)

parser.add_argument('--apply_lr_scheduling', type=int, default=1, help='1 : on, 0 : off')
parser.add_argument('--div_factor', type=float, default=10.0)
parser.add_argument('--pct_start', type=float, default=0.3)
parser.add_argument('--final_div_factor', type=float, default=10.0)

parser.add_argument('--img_aug_prob', type=float, default=0.0)
parser.add_argument('--max_rotation', type=float, default=0.0)
parser.add_argument('--max_translation', type=float, default=0.0)
parser.add_argument('--max_scale_severity', type=float, default=0.0)

def Scratch(parser):

    parser.add_argument('--bool_add_dec_header', type=int, default=1, help='if use class headers before decoding')
    parser.add_argument('--bool_learn_offset', type=int, default=0, help='if predict offsets from instance origins')
    parser.add_argument('--bool_use_vis_offset', type=int, default=0, help='if use visibility when bool_learn_offset=1')
    parser.add_argument('--bool_apply_crosshead', type=int, default=0, help='if apply cross task head')

    parser.add_argument('--obs_len', type=int, default=1)
    parser.add_argument('--pred_len', type=int, default=0)

    # Image Backbone (Intra & Inter Camera Interaction)
    parser.add_argument('--target_feat_levels', type=str, default='012', help='e.g., if it is set to 01, 0-th and 1st MS img. feat. maps are used')
    parser.add_argument('--feat_inter_repeat', type=int, default=5, help='number of TF repetition')

    # Encoder
    parser.add_argument('--num_enc_repeat', type=int, default=6, help='number of TF repetition')
    parser.add_argument('--hierarchy_depth', type=int, default=3, help='number of MS query maps')

    # Decoder
    parser.add_argument('--decoder_type', type=str, default='mask', help='mask or conv')

    args = parser.parse_args()

    return args
