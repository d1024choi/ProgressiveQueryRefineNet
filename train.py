from utils.functions import *
import argumentparser as ap
from helper import load_datasetloader, load_solvers
import torch.distributed as dist

def main(args):

    # logging setting
    logging.basicConfig(
        filename=args.save_dir + '/training.log',
        filemode="w",
        format='%(asctime)s %(levelname)s:%(message)s',
        level=logging.INFO,
        datefmt='%m/%d/%Y %I:%M:%S %p',
    )
    logger = logging.getLogger(__name__)

    consoleHandler = logging.StreamHandler(stream=sys.stdout)
    consoleHandler.setLevel(level=logging.DEBUG)
    logger.addHandler(consoleHandler)

    # DDP setting
    if (bool(args.ddp)):
        backend = 'nccl'
        dist_url = 'env://'
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])

        dist.init_process_group(backend=backend, init_method=dist_url, rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        torch.distributed.barrier()
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(args.gpu_num))
        world_size, local_rank = 1, 0

    try:
        # print training info
        if (local_rank==0): print_training_info(args, logger)

        # dtype define
        _, float_dtype = get_dtypes()

        # prepare training data (0: train, 1: valid)
        dataset, dataloader, sampler = load_datasetloader(args=args,
                                                 dtype=torch.FloatTensor,
                                                 world_size=world_size,
                                                 rank=local_rank)

        # define network
        solver = load_solvers(args, dataset[0].num_scenes, logger, float_dtype,
                              world_size=world_size, rank=local_rank)

        # torch.autograd.set_detect_anomaly(True)

        # training and validation
        for e in range(args.start_epoch, args.num_epochs):

            # ------------------------------------------
            # Training
            # ------------------------------------------
            solver.mode_selection(isTrain=True)
            if (bool(args.ddp)):
                sampler[0].set_epoch(e)
                torch.distributed.barrier()

            start = time.time()
            for b, data in enumerate(dataloader[0]):

                # debug ---
                # print(f'>> batch : {b}')
                # debug ---

                start_batch = time.time()
                solver.train(data)
                solver.learning_rate_step(e)
                end_batch = time.time()
                solver.print_training_progress(e, b, (end_batch-start_batch))

            end = time.time()
            time_left = (end - start) * (args.num_epochs - e - 1) / 3600.0

            solver.normalize_loss_tracker()
            solver.print_status(e, time_left)
            solver.init_loss_tracker()


            # ------------------------------------------
            # Evaluation
            # ------------------------------------------
            if (e % int(args.save_every) == 0):
                solver.eval(dataset[1], dataloader[1], sampler[1], e)

    except:
        logging.error(traceback.format_exc())

if __name__ == '__main__':

    # load args
    model_name = read_json(path='./config/config.json')['model_name']
    args = getattr(ap, model_name)(ap.parser)
    args.model_name = model_name

    # checks if there is a pre-defined training settings
    folder_name = f'{args.dataset_type}_{args.model_name}_model{args.exp_id}'
    args.save_dir = os.path.join('./saved_models/', folder_name)
    if args.save_dir != '' and not os.path.exists(args.save_dir):
        try: os.makedirs(args.save_dir)
        except: print(f'>> [{args.save_dir}] seems to already exist!!')

        # because there are no pre-trained nets in save_dir
        args.load_pretrained = 0

    # run main()
    main(args)


