from utils.functions import *
from utils.metrics import IoUMetric
from helper import load_datasetloader, load_solvers
from NuscenesDataset.common import CLASSES
from NuscenesDataset.visualization import BaseViz
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=int, default=534)
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--dataset_type', type=str, default='nuscenes')
    parser.add_argument('--model_name', type=str, default='Scratch')
    parser.add_argument('--is_test_all', type=int, default=1)
    parser.add_argument('--target', type=str, default='vehicle')
    parser.add_argument('--model_num', type=int, default=18)
    parser.add_argument('--visualization', type=int, default=1)
    parser.add_argument('--threshold', type=float, default=0.35)
    args = parser.parse_args()

    # logging setting
    folder_name = args.dataset_type + '_' + args.model_name + '_model' + str(args.exp_id)
    save_dir = os.path.join('./saved_models/', folder_name)
    logging.basicConfig(
        filename=save_dir + '/test.log',
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

    print_training_info(saved_args, logger)
    logger.info(f">> Test target : {args.target}")

    # load test data
    dataset, data_loader, _ = load_datasetloader(args=saved_args, dtype=torch.FloatTensor,
                                                 world_size=1, rank=0, mode='test')

    # define network
    solver = load_solvers(saved_args, dataset.num_scenes, world_size=1, rank=0, logger=logger,
                          dtype=float_dtype, isTrain=False)
    vis = BaseViz(label_indices=solver.cfg['label_indices'][args.target], SEMANTICS=CLASSES, Threshold=args.threshold)

    ckp_idx_list = read_all_saved_param_idx(solver.save_dir)

    target_models = []
    if (args.is_test_all == 0): target_models.append(args.model_num)
    else: target_models += ckp_idx_list

    for _, ckp_id in enumerate(ckp_idx_list):
        if (ckp_id not in target_models):
            logger.info(f'[SKIP] current model {ckp_id:d} is not in target model list!')
            continue

        # create empty metric
        min_visibility = 2 if args.target == 'vehicle' else None
        metric = IoUMetric(label_indices=solver.cfg['label_indices'][args.target],
                           min_visibility=min_visibility,
                           target_class=args.target) # update 231006

        # load pretrained network
        solver.load_pretrained_network_params(ckp_id)
        solver.mode_selection(isTrain=False)

        for b, batch in enumerate(tqdm(data_loader, desc='Test')):

            # if (b % 20 != 0):
            #     continue

            # inference
            target_batch = solver.reform_batch(batch, target_index=-1)
            with torch.no_grad():
                pred = solver.model(batch, float_dtype, rank=0)

            # # attention vis --------------
            # if (b == 3):
            #     attn_result = pred['attn']
            #     with open('attn_result.pickle', 'wb') as fw:
            #         pickle.dump(attn_result, fw)
            # # ----------------------------

            # visualization
            if (args.visualization == 1):
                img = np.vstack(vis(batch, pred[args.target]))

                cv2.imshow('debug', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                cv2.waitKey(0)

            # calc. IoU
            metric.update(pred[args.target][0], target_batch)

        results = metric.compute()
        max_key = max(results, key=results.get)
        max_val = results[max_key]
        logger.info(f">> (TEST@CKPID {ckp_id}) {max_key} : {max_val:.3f}")


if __name__ == '__main__':
    main()


