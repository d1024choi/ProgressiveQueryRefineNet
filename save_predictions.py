from utils.functions import *
from utils.metrics import IoUMetric
from helper import load_datasetloader, load_solvers
from NuscenesDataset.common import CLASSES
from NuscenesDataset.save_pred import BaseSave
import matplotlib.pyplot as plt
import dill

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=int, default=624)
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--dataset_type', type=str, default='nuscenes')
    parser.add_argument('--model_name', type=str, default='Scratch')
    parser.add_argument('--is_test_all', type=int, default=1)
    parser.add_argument('--target', type=str, default='lane')
    parser.add_argument('--model_num', type=int, default=24)
    parser.add_argument('--save_original_images', type=int, default=0)
    parser.add_argument('--threshold', type=float, default=0.4)
    args = parser.parse_args()


    # logging setting
    folder_name = args.dataset_type + '_' + args.model_name + '_model' + str(args.exp_id)
    save_dir = os.path.join('./saved_models/', folder_name)
    logging.basicConfig(
        filename=save_dir + '/pred_save.log',
        filemode="w",
        format='%(asctime)s %(levelname)s:%(message)s',
        level=logging.INFO,
        datefmt='%m/%d/%Y %I:%M:%S %p',
    )

    logger = logging.getLogger(__name__)
    consoleHandler = logging.StreamHandler(stream=sys.stdout)
    consoleHandler.setLevel(level=logging.DEBUG)
    logger.addHandler(consoleHandler)

    # run train.py
    try: test(args, logger)
    except: logging.error(traceback.format_exc())

def test(args, logger):

    # CUDA setting
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(args.gpu_num))

    # type definition
    _, float_dtype = get_dtypes(useGPU=True)

    # path to saved network
    folder_name = args.dataset_type + '_' + args.model_name + '_model' + str(args.exp_id)
    path = os.path.join('./saved_models/', folder_name)

    # load parameter setting
    with open(os.path.join(path, 'config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)
    saved_args.ddp = 0
    saved_args.bool_mixed_precision = 0
    saved_args.save_dir = path
    saved_args.exp_id = args.exp_id

    # save folder
    folder_name = args.model_name + '_exp' + str(args.exp_id) + '_m' + str(args.model_num)
    save_dir = os.path.join('./VisResults/', folder_name)
    if save_dir != '' and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print_training_info(saved_args, logger)
    logger.info(f">> Test target : {args.target}")

    # load test data
    dataset, data_loader, _ = load_datasetloader(args=saved_args, dtype=torch.FloatTensor,
                                                 world_size=1, rank=0, mode='test')

    # define network
    solver = load_solvers(saved_args, dataset.num_scenes, world_size=1, rank=0, logger=logger,
                          dtype=float_dtype, isTrain=False)
    save = BaseSave(label_indices=solver.cfg['label_indices'], SEMANTICS=CLASSES, threshold=args.threshold)


    # load pretrained network
    solver.load_pretrained_network_params(args.model_num)
    solver.mode_selection(isTrain=False)

    for b, batch in enumerate(tqdm(data_loader, desc='Test')):

        if (b > 500): continue

        # inference
        with torch.no_grad():
            pred = solver.model(batch, float_dtype, rank=0)


        # save
        if (args.save_original_images == 1):
            img = save.return_cams(batch)
            file_name_cam = 'cam_%04d.ckpl' % b
            file_path = os.path.join(save_dir, file_name_cam)
            with open(file_path, 'wb') as f:
                dill.dump(img, f, protocol=dill.HIGHEST_PROTOCOL)


        gt_bev = save.return_gt(batch, args.target)
        pred_bev = save.return_pred(pred, args.target)
        aux_bev = save.return_pred(pred, 'intm')
        data = {'gt': gt_bev, 'pred': pred_bev, 'aux': aux_bev}

        file_name_bev = args.target + '_%04d.ckpl' % b
        file_path = os.path.join(save_dir, file_name_bev)
        with open(file_path, 'wb') as f:
            dill.dump(data, f, protocol=dill.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()


