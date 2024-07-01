import cv2
from utils.functions import *
from utils.metrics import IoUMetric
from helper import load_datasetloader, load_solvers
from NuscenesDataset.common import CLASSES
from NuscenesDataset.save_pred import BaseSave
import matplotlib.pyplot as plt
import dill

COLORS = {
    # static
    'drivable': (110, 110, 110),


    # dividers
    'laneline': (0, 0, 255),

    # dynamic
    'vehicle': (0, 158, 255),
    'pedestrian': (230, 0, 0),

    'nothing': (200, 200, 200)
}

def main():


    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=int, default=534)
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--dataset_type', type=str, default='nuscenes')
    parser.add_argument('--model_name', type=str, default='Scratch')
    parser.add_argument('--model_num', type=int, default=27)
    parser.add_argument('--start_idx', type=int, default=2)
    parser.add_argument('--end_idx', type=int, default=200)
    args = parser.parse_args()

    # save folder
    folder_name = args.model_name + '_exp' + str(args.exp_id) + '_m' + str(args.model_num)
    save_dir = os.path.join('./VisResults/', folder_name)
    if os.path.exists(save_dir) == False:
        sys.exit(f'>> no such files or directories...{save_dir}')

    for idx in range(args.start_idx, args.end_idx):

        # cam data ---
        cam_data = load_file(save_dir, 'cam', idx)
        img = draw_cam_data(cam_data)

        file_name = './VisResults/cam_%04d.png' % idx
        cv2.imwrite(file_name, img.astype('uint8'))

        # bev data ---
        veh_data = load_file(save_dir, 'vehicle', idx)
        road_data = load_file(save_dir, 'road', idx)
        ped_data = load_file(save_dir, 'pedestrian', idx)
        line_data = load_file(save_dir, 'lane', idx)

        bev_gt = draw_bev(veh_data, road_data, ped_data, line_data, target='gt')
        bev_pred = draw_bev(veh_data, road_data, ped_data, line_data, target='pred')
        bev_aux = draw_bev(veh_data, road_data, ped_data, line_data, target='aux')

        bev = np.hstack([bev_gt, bev_aux, bev_pred])
        # cv2.imshow("", cv2.cvtColor(bev, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)

        file_name = './VisResults/bev_%04d.png' % idx
        cv2.imwrite(file_name, bev.astype('uint8'))



def load_file(save_dir, target, idx):
    file_name = target + '_%04d.ckpl' % idx
    file_path = os.path.join(save_dir, file_name)
    if (os.path.exists(file_path)):
        with open(file_path, 'rb') as f:
            return dill.load(f, encoding='latin1')
    return None

def draw_cam_data(cam_data):
    '''
    cam_data : b n_cams h w ch
    img : 2*h 3*w ch
    '''

    if (cam_data is None):
        return None

    upper = np.hstack([cv2.cvtColor(cam_data[0, 0], cv2.COLOR_RGB2BGR),
                       cv2.cvtColor(cam_data[0, 1], cv2.COLOR_RGB2BGR),
                       cv2.cvtColor(cam_data[0, 2], cv2.COLOR_RGB2BGR)])

    lower = np.hstack([cv2.cvtColor(cam_data[0, 3], cv2.COLOR_RGB2BGR),
                       cv2.cvtColor(cam_data[0, 4], cv2.COLOR_RGB2BGR),
                       cv2.cvtColor(cam_data[0, 5], cv2.COLOR_RGB2BGR)])

    img = np.vstack([upper, lower])

    # cv2.imshow("", img)
    # cv2.waitKey(0)
    return img

def draw_bev(veh, road, ped, line, target='gt'):

    '''
    result : h w 3
    '''

    zeros = np.zeros(shape=(200, 200, 1))

    vehicle = np.copy(zeros)
    if (veh is not None):
        vehicle = torch.from_numpy(veh[target]).permute(1, 2, 0).numpy()

    drivable = np.copy(zeros)
    if (road is not None):
        drivable = torch.from_numpy(road[target]).permute(1, 2, 0).numpy()

    laneline = np.copy(zeros)
    if (line is not None):
        laneline = torch.from_numpy(line[target]).permute(1, 2, 0).numpy()

    pedestrian = np.copy(zeros)
    if (ped is not None):
        pedestrian = torch.from_numpy(ped[target]).permute(1, 2, 0).numpy()

    bev = np.concatenate([drivable, laneline, vehicle, pedestrian], axis=-1)

    h, w, c = bev.shape

    # Prioritize higher class labels
    eps = (1e-5 * np.arange(c))[None, None]  # 1 1 c
    idx = (bev + eps).argmax(axis=-1)  # h w
    val = np.take_along_axis(bev, idx[..., None], -1)

    # Spots with no labels are light grey
    empty = np.uint8(COLORS['nothing'])[None, None]  # 1 1 3

    colors = get_colors(['drivable', 'laneline', 'vehicle', 'pedestrian', 'nothing'])
    result = (val * colors[idx]) + ((1 - val) * empty)
    result = np.uint8(result)


    # cv2.imshow("", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)

    result = np.pad(result, pad_width=((10, 10), (10, 10), (0, 0)), constant_values=255)

    return result

def get_colors(semantics):
    return np.array([COLORS[s] for s in semantics], dtype=np.uint8)


if __name__ == '__main__':
    main()


