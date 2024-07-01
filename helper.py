from torch.utils.data import DataLoader
from utils.functions import *
from torch.utils.data.distributed import DistributedSampler

LOADER_TYPE_Apro = ['Scratch']

def load_datasetloader(args, dtype, world_size, rank, mode='train'):

    config = read_json(path='./config/config.json')

    if (args.dataset_type == 'nuscenes'):
        if (args.model_name in LOADER_TYPE_Apro):
            from NuscenesDataset.loader_typeApro import DatasetLoader
        else:
            sys.exit("[Error] No loader type exists for '%s' in 'Nuscenes' !!" % args.model_name)
        args.dataset_dir = config['nuscenes']['dataset_dir']
    else:
        sys.exit("[Error] '%s' dataset is not supported !!" % args.dataset_type)


    if (mode in ['train', 'val', 'valid']):
        if (bool(args.ddp)):
            train_dataset = DatasetLoader(args=args, dtype=dtype, world_size=world_size, rank=rank, mode='train')
            val_dataset = DatasetLoader(args=args, dtype=dtype, world_size=world_size, rank=rank, mode='val')

            train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
            val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_cores, pin_memory=True, sampler=train_sampler)

            val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_cores, pin_memory=True, sampler=val_sampler)
        else:
            train_dataset = DatasetLoader(args=args, dtype=dtype, world_size=1, rank=0, mode='train')
            val_dataset = DatasetLoader(args=args, dtype=dtype, world_size=1, rank=0, mode='val')

            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_cores, drop_last=True)
            val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_cores, drop_last=True)

            train_sampler, val_sampler = None, None

        return (train_dataset, val_dataset), (train_dataloader, val_dataloader), (train_sampler, val_sampler)
    else:
        test_dataset = DatasetLoader(args=args, dtype=dtype, world_size=1, rank=0, mode='test')
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_cores, drop_last=False)
        return test_dataset, test_dataloader, None


def load_solvers(args, num_train_scenes, logger, dtype, world_size=None, rank=None, isTrain=True):

    if (args.model_name == 'Scratch'):
        from optimization.Scratch_solver import Solver
        return Solver(args, num_train_scenes, world_size, rank, logger, dtype, isTrain)
    else:
        sys.exit("[Error] There is no solver for '%s' !!" % args.model_name)
