from utils.functions import *
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from torch.utils.data import Dataset
from utils.geometry import *
from NuscenesDataset.common import *
from torchvision import transforms
from nuscenes.utils.data_classes import Box
from shapely.geometry import MultiPolygon
from utils.augmentation import PhotoMetricDistortion, AffineTransform
import matplotlib.pyplot as plt

class DatasetLoader(Dataset):

    def __init__(self, args, dtype, world_size=None, rank=None, mode='train'):

        random.seed(1024)

        self.mode = mode
        split = 'train' if (mode in ['train', 'val', 'valid']) else 'val'

        self.args = args
        self.cfg = config_update(read_config(model_name=args.model_name), args)
        self.seq_len = args.obs_len + args.pred_len

        # Image resize and crop
        ori_dims = (self.cfg['original_image']['w'], self.cfg['original_image']['h'])
        resize_dims = (self.cfg['image']['w'], self.cfg['image']['h']+self.cfg['image']['top_crop'])
        crop = (0, self.cfg['image']['top_crop'], resize_dims[0], resize_dims[1])
        self.img_aug_params = {'scale_width': resize_dims[0]/ori_dims[0],
                               'scale_height': resize_dims[1]/ori_dims[1],
                               'resize_dims': resize_dims,
                               'crop': crop}

        # Normalising input images
        self.img_norm = transforms.Compose([transforms.ToTensor()])

        aug_params = self.cfg['img_augmentation']
        self.img_distortion = AffineTransform(max_rotation_degree=aug_params['max_rotation'],
                                              max_translation_pixel=aug_params['max_translation'],
                                              max_scale_severity=aug_params['max_scale_severity'])

        # Bird's-eye view parameters
        self.bev_resolution, self.bev_start_position, self.bev_dimension = \
            calculate_birds_eye_view_parameters(self.cfg['lift']['x_bound'],
                                                self.cfg['lift']['y_bound'],
                                                self.cfg['lift']['z_bound'],
                                                isnumpy=True)


        # Nuscenes
        self.nusc_map = {}
        for k, v in enumerate(MAP_NAMES):
            self.nusc_map.update({v: NuScenesMap(dataroot=self.cfg['nuscenes']['dataset_dir'], map_name=v)})

        self.nusc = NuScenes(version=self.cfg['nuscenes']['version'],
                             dataroot=self.cfg['nuscenes']['dataset_dir'], verbose=False)


        # Splits into train/valid/test
        self.split_scene = self.get_split(split)
        self.sample_records = self.return_ordered_sample_records()
        seq_sample_indices = self.return_seq_sample_indices()
        random.shuffle(seq_sample_indices)

        if (mode in ['train', 'val', 'valid']):
            num_val_scenes = int(len(seq_sample_indices) * self.args.val_ratio)
            num_train_scenes = len(seq_sample_indices) - num_val_scenes
            train_scenes = seq_sample_indices[:num_train_scenes]
            val_scenes = []
            for r in range(world_size):
                val_scenes += seq_sample_indices[num_train_scenes:]
            random.shuffle(val_scenes)
            self.scenes = train_scenes if (mode == 'train') else val_scenes
        else:
            self.scenes = seq_sample_indices
        self.num_scenes = len(self.scenes)

        if (rank==0):
            print(">> Dataset is loaded from {%s} " % os.path.basename(__file__))
            print(f'>> Number of available {mode} samples is {self.num_scenes}')

    def __len__(self):
        return self.num_scenes

    def __getitem__(self, idx):
        seq_indices = self.scenes[idx]
        seq_indices = np.array(seq_indices)[self.cfg['target_frame_indices']]
        data = self.extract_seqdata_from_sample_records(seq_indices)
        return data

    def next_sample(self, seq_index):

        seq_indices = self.scenes[seq_index]
        seq_indices = np.array(seq_indices)[self.cfg['target_frame_indices']]
        data = self.extract_seqdata_from_sample_records(seq_indices)
        return data

    def extract_seqdata_from_sample_records(self, seq_indices):

        '''
        BEV images are flipped upside-down and left-right.
        Need to apply np.flipud(np.fliplr(bev)) in order to match ego-centric frame (forward-up, side-left)
        '''

        instance_map = {}
        seq_images, seq_intrinsics, seq_extrinsics, seq_bev, seq_center, seq_visibility, \
            seq_instance, seq_offsets, seq_records, seq_e2w, seq_w2e = [], [], [], [], [], [], [], [], [], [], []

        for _, i in enumerate(seq_indices):
            rec = self.sample_records[i]
            seq_records.append(rec)

            '''
            images (1 x n x c x h x w, tensor, float32) : normalized to 0~1 and (maybe) by mean and var
            intrinsics (1 x n x 3 x 3, tensor, float32) : crop and scaling are reflect
            extrinsics (1 x n x 4 x 4, tensor, float32) : egolidar_to_camera
            bev (1 x 12 x h x w, tensor, float32)
            center (1 x 1 x h x w, tensor, float32)
            visibility (1 x 1 x h x w, tensor, uint8)
            w2e, e2w (1 x 4 x 4, tensor, float32)
            '''

            # extrinsics : egolidar_to_camera
            images, intrinsics, extrinsics, e2w, w2e = self.return_input_data(rec)
            seq_e2w.append(e2w[None])
            seq_w2e.append(w2e[None])

            data, instance_map = self.return_bev_labels(rec, instance_map)
            bev = data['bev']
            center = torch.cat((data['aux']['center_score_veh'], data['aux']['center_score_ped']), dim=1) # 1 2 h w
            instance = torch.cat((data['aux']['instance_veh'], data['aux']['instance_ped']), dim=1)  # 1 2 h w
            offset = torch.cat((data['aux']['center_offset_veh'], data['aux']['center_offset_ped']), dim=1)  # 1 4 h w
            visibility = data['visibility']

            # flip
            bev = torch.flip(bev, dims=(2, 3))
            center = torch.flip(center, dims=(2, 3))
            visibility = torch.flip(visibility, dims=(2, 3))
            instance = torch.flip(instance, dims=(2, 3))
            offset = torch.flip(offset, dims=(2, 3))

            seq_images.append(images)
            seq_intrinsics.append(intrinsics)
            seq_extrinsics.append(extrinsics)
            seq_bev.append(bev)
            seq_center.append(center)
            seq_visibility.append(visibility)
            seq_instance.append(instance)
            seq_offsets.append(offset)

        seq_images = torch.cat(seq_images, dim=0)
        seq_intrinsics = torch.cat(seq_intrinsics, dim=0)
        seq_extrinsics = torch.cat(seq_extrinsics, dim=0)
        seq_bev = torch.cat(seq_bev, dim=0)
        seq_center = torch.cat(seq_center, dim=0)
        seq_instance = torch.cat(seq_instance, dim=0)
        seq_visibility = torch.cat(seq_visibility, dim=0)
        seq_offsets = torch.cat(seq_offsets, dim=0)
        seq_e2w = torch.cat(seq_e2w, dim=0)
        seq_w2e = torch.cat(seq_w2e, dim=0)

        data = {'image': seq_images,
                'intrinsics': seq_intrinsics,
                'extrinsics': seq_extrinsics,
                'bev': seq_bev,
                'center': seq_center,
                'instance': seq_instance,
                'visibility': seq_visibility,
                'offsets': seq_offsets,
                # 'seq_records': seq_records,
                'e2w' : seq_e2w,
                'w2e' : seq_w2e}


        return data

    def return_ordered_sample_records(self):
        '''
        Based on 'https://github.com/wayveai/fiery'
        '''

        samples = [samp for samp in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [samp for samp in samples if self.nusc.get('scene', samp['scene_token'])['name'] in self.split_scene]

        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return samples

    def return_seq_sample_indices(self):
        '''
        Based on 'https://github.com/wayveai/fiery'
        '''

        indices = []
        for index in range(len(self.sample_records)):
            is_valid_data = True
            previous_rec = None
            current_indices = []
            for t in range(self.seq_len):
                index_t = index + t
                # Going over the dataset size limit.
                if index_t >= len(self.sample_records):
                    is_valid_data = False
                    break
                rec = self.sample_records[index_t]
                # Check if scene is the same
                if (previous_rec is not None) and (rec['scene_token'] != previous_rec['scene_token']):
                    is_valid_data = False
                    break

                current_indices.append(index_t)
                previous_rec = rec

            if is_valid_data:
                indices.append(current_indices)

        return indices

    def get_split(self, split):
        split_dir = Path(__file__).parent / 'nuscenes/splits'
        split_path = split_dir / f'{split}.txt'
        return split_path.read_text().strip().split('\n')

    def return_input_data(self, sample_record):
        '''
        egolidar : ego-pose based on lidar sensor
        egocam : ego-pose based on camera sensor
        '''

        lidar_record = self.nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
        egolidar = self.nusc.get('ego_pose', lidar_record['ego_pose_token'])
        top_crop, left_crop = self.img_aug_params['crop'][1], self.img_aug_params['crop'][0]

        # note : Fiery sets 'flat=True'
        EL2W = get_pose(egolidar['rotation'], egolidar['translation'], flat=True) # egolidar_to_world
        W2EL = get_pose(egolidar['rotation'], egolidar['translation'], flat=True, inv=True)  # egolidar_to_world

        images, intrinsics, extrinsics = [], [], []
        for cam in CAMERAS:

            cam_token = sample_record['data'][cam]
            cam_record = self.nusc.get('sample_data', cam_token)
            egocam = self.nusc.get('ego_pose', cam_record['ego_pose_token'])
            cam = self.nusc.get('calibrated_sensor', cam_record['calibrated_sensor_token'])

            # Intrinsic parameter
            intrinsic = torch.from_numpy(np.array(cam['camera_intrinsic'])) # 3 x 3
            intrinsic = update_intrinsics(intrinsic, top_crop, left_crop,
                                          scale_width=self.img_aug_params['scale_width'],
                                          scale_height=self.img_aug_params['scale_height'])

            # Extrinsic parameter
            EC2C = get_pose(cam['rotation'], cam['translation'], flat=False, inv=True) # egocam_to_cam
            extrinsic = torch.from_numpy(EC2C)  # 4 x 4

            # note: Fiery uses C2EL, no dramatic performance improvement with EL2C
            # W2EC = get_pose(egocam['rotation'], egocam['translation'], flat=False, inv=True) # world_to_egocam
            # EL2C = EC2C @ W2EC @ EL2W # egolidar_to_cam
            # extrinsic = torch.from_numpy(EL2C) # 4 x 4


            # Surround Image
            img = Image.open(Path(self.nusc.get_sample_data_path(cam_token)))
            img = resize_and_crop_image(img, resize_dims=self.img_aug_params['resize_dims'],
                                        crop=self.img_aug_params['crop'])

            if (self.mode == 'train' and np.random.rand(1) < self.cfg['img_aug_prob']):
                img = self.img_distortion(img, intrinsic)

            img = self.img_norm(img) # c x h x w

            images.append(img.unsqueeze(0).unsqueeze(0))
            intrinsics.append(intrinsic.unsqueeze(0).unsqueeze(0))
            extrinsics.append(extrinsic.unsqueeze(0).unsqueeze(0))

        images = torch.cat(images, dim=1)
        intrinsics = torch.cat(intrinsics, dim=1)
        extrinsics = torch.cat(extrinsics, dim=1)

        return images, intrinsics, extrinsics, torch.from_numpy(EL2W), torch.from_numpy(W2EL)

    def return_bev_labels(self, sample_record, instance_map):
        '''
        Based on https://github.com/bradyz/cross_view_transformers
        '''

        scene_token = sample_record['scene_token']
        scene_record = self.nusc.get('scene', scene_token)
        location = self.nusc.get('log', scene_record['log_token'])['location']
        lidar_sample = self.nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
        egopose = self.nusc.get('ego_pose', lidar_sample['ego_pose_token'])

        # Raw annotations
        anns_dynamic = self.get_ann_rec_by_category(sample_record, DYNAMIC)

        # BEV images
        static = self.get_static_layers(location, egopose, STATIC)    # 200 x 200 x 2
        dividers = self.get_line_layers(location, egopose, DIVIDER)   # 200 x 200 x 2
        dynamic = self.get_dynamic_layers(anns_dynamic, egopose)      # 200 x 200 x 8
        bev = np.concatenate((static, dividers, dynamic), -1)         # 200 x 200 x 12

        # Data for auxillary tasks,
        anns_dynamic_all = []
        for anns_list in anns_dynamic: anns_dynamic_all += anns_list
        _aux, visibility_veh, visibility_ped, instance_map = self.get_dynamic_objects(anns_dynamic_all, egopose, instance_map)


        bev = torch.from_numpy(bev).permute(2, 0, 1).unsqueeze(0).contiguous() # 1 x 1 x 12 x h x w, float
        aux = {}
        for key, value in _aux.items():
            aux[key] = torch.from_numpy(value).permute(2, 0, 1).unsqueeze(0).contiguous() # 1 x 1 x c x h x w

        visibility_veh = torch.from_numpy(visibility_veh).unsqueeze(0).unsqueeze(0) # 1 x 1 x h x w, float
        visibility_ped = torch.from_numpy(visibility_ped).unsqueeze(0).unsqueeze(0)  # 1 x 1 x h x w, float
        visibility = torch.cat((visibility_veh, visibility_ped), dim=1)

        data = {'bev': bev,
                'aux': aux,
                'visibility': visibility,
                'location': location,
                'egopose': egopose}

        return data, instance_map

    def get_ann_rec_by_category(self, sample, categories):
        result = [[] for _ in categories]

        for ann_token in self.nusc.get('sample', sample['token'])['anns']:
            a = self.nusc.get('sample_annotation', ann_token)
            idx = self.get_category_index(a['category_name'], categories)

            if idx is not None:
                result[idx].append(a)

        return result

    def get_category_index(self, name, categories):
        """
        human.pedestrian.adult
        """
        tokens = name.split('.')

        for i, category in enumerate(categories):
            if category in tokens:
                return i

        return None

    def get_dynamic_layers(self, anns_by_category, egopose):

        # egopose (lidar)
        trans = -np.array(egopose['translation'])
        yaw = Quaternion(egopose['rotation']).yaw_pitch_roll[0]
        rot = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse

        # bev center/resolution
        bev_center = - self.bev_start_position[:2] + 0.5 * self.bev_resolution[:2]
        bev_res = self.bev_resolution[:2]

        result = list()
        for anns in anns_by_category:
            render = np.zeros((self.cfg['bev']['h'], self.cfg['bev']['w']), dtype=np.uint8)
            for ann in anns:
                box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))
                box.translate(trans)
                box.rotate(rot)

                pts = box.bottom_corners()[:2].T
                pts = np.round((pts + bev_center) / bev_res).astype(np.int32)
                pts[:, [1, 0]] = pts[:, [0, 1]]
                cv2.fillPoly(render, [pts], 1.0, INTERPOLATION)
            result.append(render)

        return np.stack(result, -1).astype('float32')

    def get_static_layers(self, location, egopose, layers, patch_radius=150):

        # egopose
        trans = -np.array(egopose['translation'])[:2]
        yaw = Quaternion(egopose['rotation']).yaw_pitch_roll[0]
        rot = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse.rotation_matrix[:2, :2]

        # bev center/resolution
        bev_center = - self.bev_start_position[:2] + 0.5 * self.bev_resolution[:2]
        bev_res = self.bev_resolution[:2]

        pose = get_pose(egopose['rotation'], egopose['translation'], flat=True)
        x, y = pose[0][-1], pose[1][-1]
        box_coords = (x - patch_radius, y - patch_radius, x + patch_radius, y + patch_radius)
        records_in_patch = self.nusc_map[location].get_records_in_patch(box_coords, layers, 'intersect')

        result = list()

        for layer in layers:
            render = np.zeros((self.cfg['bev']['h'], self.cfg['bev']['w']), dtype=np.uint8)

            for r in records_in_patch[layer]:
                polygon_token = self.nusc_map[location].get(layer, r)

                if layer == 'drivable_area': polygon_tokens = polygon_token['polygon_tokens']
                else: polygon_tokens = [polygon_token['polygon_token']]

                for p in polygon_tokens:
                    polygon = self.nusc_map[location].extract_polygon(p)
                    polygon = MultiPolygon([polygon])

                    exteriors = [np.array(poly.exterior.coords).T for poly in polygon.geoms] # 2 x N
                    exteriors = [rot @ (p.T + trans).T for p in exteriors] # 2 x N
                    exteriors = [np.round((p.T + bev_center) / bev_res).astype(np.int32) for p in exteriors] # N x 2
                    exteriors = [np.fliplr(p) for p in exteriors]  # N x 2

                    cv2.fillPoly(render, exteriors, 1, INTERPOLATION)

                    interiors = [np.array(pi.coords).T for poly in polygon.geoms for pi in poly.interiors]
                    interiors = [rot @ (p.T + trans).T for p in interiors]
                    interiors = [np.round((p.T + bev_center) / bev_res).astype(np.int32) for p in interiors]
                    interiors = [np.fliplr(p) for p in interiors]  # N x 2

                    cv2.fillPoly(render, interiors, 0, INTERPOLATION)

            result.append(render)
        return np.stack(result, -1).astype('float32')

    def get_line_layers(self, location, egopose, layers, patch_radius=150, thickness=1):

        # egopose
        trans = -np.array(egopose['translation'])[:2]
        yaw = Quaternion(egopose['rotation']).yaw_pitch_roll[0]
        rot = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse.rotation_matrix[:2, :2]

        # bev center/resolution
        bev_center = - self.bev_start_position[:2] + 0.5 * self.bev_resolution[:2]
        bev_res = self.bev_resolution[:2]

        pose = get_pose(egopose['rotation'], egopose['translation'], flat=True)
        x, y = pose[0][-1], pose[1][-1]
        box_coords = (x - patch_radius, y - patch_radius, x + patch_radius, y + patch_radius)
        records_in_patch = self.nusc_map[location].get_records_in_patch(box_coords, layers, 'intersect')

        result = list()

        for layer in layers:
            render = np.zeros((self.cfg['bev']['h'], self.cfg['bev']['w']), dtype=np.uint8)

            for r in records_in_patch[layer]:
                polygon_token = self.nusc_map[location].get(layer, r)
                line = self.nusc_map[location].extract_line(polygon_token['line_token'])

                p = np.float32(line.xy)    # 2 x N
                p = rot @ (p.T + trans).T  # 2 x N
                p = np.round((p.T + bev_center) / bev_res).astype(np.int32) # N x 2
                p = np.fliplr(p) # N x 2

                cv2.polylines(render, [p], False, 1, thickness=thickness)

            result.append(render)

        return np.stack(result, -1).astype(np.float32)

    def get_dynamic_objects(self, anns, egopose, ins_map):

        # egopose
        trans = -np.array(egopose['translation'])
        yaw = Quaternion(egopose['rotation']).yaw_pitch_roll[0]
        rot = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse

        # bev center/resolution
        h, w = self.cfg['bev']['h'], self.cfg['bev']['w']
        bev_center = - self.bev_start_position[:2] + 0.5 * self.bev_resolution[:2]
        bev_res = self.bev_resolution[:2]

        segmentation = np.zeros((h, w), dtype=np.uint8)
        center_ohw = np.zeros((h, w, 4), dtype=np.float32)

        center_score_veh = np.zeros((h, w), dtype=np.float32)
        center_score_ped = np.zeros((h, w), dtype=np.float32)
        center_offset_veh = np.zeros((h, w, 2), dtype=np.float32)
        center_offset_ped = np.zeros((h, w, 2), dtype=np.float32)
        visibility_veh = np.full((h, w), 255, dtype=np.uint8)
        visibility_ped = np.full((h, w), 255, dtype=np.uint8)
        instance_veh = np.zeros((h, w), dtype=np.uint8)
        instance_ped = np.zeros((h, w), dtype=np.uint8)

        sigma = 1
        buf = np.zeros((h, w), dtype=np.uint8)
        coords = np.stack(np.meshgrid(np.arange(w), np.arange(h)), -1).astype(np.float32)
        for ann in anns:
            box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))
            box.translate(trans)
            box.rotate(rot)

            p = box.bottom_corners()[:2].T
            p = np.round((p + bev_center) / bev_res).astype(np.int32)
            p[:, [1, 0]] = p[:, [0, 1]] # 4 x 2

            center = np.round((box.center[:2] + bev_center) / bev_res).astype(np.int32).reshape(1, 2)
            center = np.fliplr(center)

            buf.fill(0)
            cv2.fillPoly(buf, [p], 1, INTERPOLATION)
            mask = buf > 0

            if not np.count_nonzero(mask):
                continue

            # segmentation up
            segmentation[mask] = 255

            # instance map up
            if ann['instance_token'] not in ins_map:
                ins_map[ann['instance_token']] = len(ins_map) + 1
            ins_id = ins_map[ann['instance_token']]

            if ('vehicle' in ann['category_name']):
                visibility_veh[mask] = ann['visibility_token']
                instance_veh[mask] = ins_id
                center_offset_veh[mask] = center - coords[mask]
                center_score_veh[mask] = np.exp(-(center_offset_veh[mask] ** 2).sum(-1) / (sigma ** 2))
            elif ('pedestrian' in ann['category_name']):
                visibility_ped[mask] = ann['visibility_token']
                instance_ped[mask] = ins_id
                center_offset_ped[mask] = center - coords[mask]
                center_score_ped[mask] = np.exp(-(center_offset_ped[mask] ** 2).sum(-1) / (sigma ** 2))

        segmentation = np.float32(segmentation[..., None])
        center_score_veh = center_score_veh[..., None]
        center_score_ped = center_score_ped[..., None]
        instance_veh = instance_veh[..., None]
        instance_ped = instance_ped[..., None]

        result = {'segmentation': segmentation,
                  'center_ohw': center_ohw,
                  'center_score_veh': center_score_veh,
                  'center_score_ped': center_score_ped,
                  'center_offset_veh': center_offset_veh,
                  'center_offset_ped': center_offset_ped,
                  'instance_veh': instance_veh,
                  'instance_ped': instance_ped}

        return result, visibility_veh, visibility_ped, ins_map

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--val_ratio', type=float, default=0.05)
    parser.add_argument('--h', type=int, default=224)
    parser.add_argument('--w', type=int, default=480)
    parser.add_argument('--top_crop', type=int, default=46)
    args = parser.parse_args()

    from visualization import BaseViz
    vis = BaseViz(label_indices=[1, 2, 3, 4], SEMANTICS=CLASSES)
    loader = DatasetLoader(args=args, dtype=None)


    for s in range(loader.num_val_scenes):
        data = loader.next_sample(s, mode='val')
        img = np.vstack(vis(data))

        cv2.imshow('debug', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)


if __name__ == '__main__':
    main()