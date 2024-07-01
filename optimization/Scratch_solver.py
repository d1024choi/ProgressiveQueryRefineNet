import os.path

from utils.functions import *
from utils.loss import *
from utils.metrics import IoUMetric
from models.scratch.scratch import Scratch
from torch.nn.parallel import DistributedDataParallel as DDP


class Solver:

    def __init__(self, args, num_train_scenes, world_size=None, rank=None, logger=None, dtype=None, isTrain=True):

        # save folder path
        folder_name = f'{args.dataset_type}_{args.model_name}_model{args.exp_id}'
        self.save_dir = os.path.join('./saved_models/', folder_name)

        # load pre-trained settings or save current settings
        if (isTrain):
            if (args.load_pretrained == 1):
                if (os.path.exists(self.save_dir) is not True): sys.exit(f'>> path {self.save_dir} does not exist!!')
                else:
                    with open(os.path.join(self.save_dir, 'config.pkl'), 'rb') as f:
                        args = pickle.load(f)

            # or save current settings
            else:
                if (rank==0):
                    with open(os.path.join(self.save_dir, 'config.pkl'), 'wb') as f:
                        pickle.dump(args, f)

        # training setting
        self.args = args
        self.rank, self.world_size = rank, world_size
        if (isTrain):
            self.cfg = config_update(read_config(model_name=args.model_name), args)
            with open(os.path.join(self.save_dir, 'config_dict.pkl'), 'wb') as f:
                pickle.dump(self.cfg, f)
        else:
            if (os.path.exists(os.path.join(self.save_dir, 'config_dict.pkl'))):
                with open(os.path.join(self.save_dir, 'config_dict.pkl'), 'rb') as f:
                    self.cfg = pickle.load(f)
            else:
                self.cfg = config_update(read_config(model_name=args.model_name), args)

        self.log = logger
        self.target = self.cfg['target']
        self.dtype = dtype
        self.num_batches = int(num_train_scenes / (args.batch_size * world_size))
        self.total_num_iteration = args.num_epochs * self.num_batches

        # training monitoring
        self.monitor = {'iter': 0,
                        'total_loss': 0,
                        'bce': 0,
                        'prev_IoU': 0,
                        'cur_lr': self.args.learning_rate,
                        'min_lr': self.args.min_learning_rate}

        # model define
        model = Scratch(self.cfg)
        if (rank == 0):
            num_model_params = float(self.count_parameters(model)) / 1e6
            print(">> Number of params of this model is %.1f M" % num_model_params)


        if (bool(args.ddp)):
            model.type(dtype).to(rank)
            self.model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        else:
            self.model = model.type(dtype).cuda()


        # for mixed precision training
        self.scaler = torch.cuda.amp.GradScaler()

        # define optimizer
        self.opt = optim.AdamW(self.model.parameters(), lr=args.learning_rate)

        # define loss and metric
        min_visibility = None if self.cfg['target']=='road' else 2
        self.loss = LossScratch(cfg=self.cfg, min_visibility=min_visibility)

        # training schedule
        self.lr_scheduler = optim.lr_scheduler.OneCycleLR(self.opt,
                                                          max_lr=self.args.learning_rate,
                                                          div_factor=self.args.div_factor, # starts at max_lr / 10
                                                          final_div_factor=self.args.final_div_factor, # ends at lr / 10 / 10
                                                          pct_start=self.args.pct_start, # reaches max_lr at 30% of total steps
                                                          steps_per_epoch=self.num_batches,
                                                          epochs=self.args.num_epochs,
                                                          cycle_momentum=False)

        # load network params
        if (args.load_pretrained == 1):
            ckp_idx = save_read_latest_checkpoint_num(os.path.join(self.save_dir), 0, isSave=False)
            _ = self.load_pretrained_network_params(ckp_idx)

        if (rank == 0):
            print(">> Optimizer is loaded from {%s} " % os.path.basename(__file__))

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def mode_selection(self, isTrain=True):
        if (isTrain): self.model.train()
        else: self.model.eval()

    def init_loss_tracker(self):
        self.monitor['bce'] = 0


    def normalize_loss_tracker(self):
        self.monitor['bce'] /= self.num_batches

    def learning_rate_step(self, e=None):
        if (bool(self.args.apply_lr_scheduling)):
            self.lr_scheduler.step()

    def load_pretrained_network_params(self, ckp_idx, isTrain=False):

        file_name = self.save_dir + '/saved_chk_point_%d.pt' % ckp_idx
        checkpoint = torch.load(file_name, map_location=torch.device('cpu'))

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            name = k.replace("module.", "") # removing ‘moldule.’ from key
            new_state_dict[name] = v

        pretrained_dict = {k: v for k, v in new_state_dict.items() if k in self.model.state_dict()}
        self.model.load_state_dict(pretrained_dict)
        self.monitor['iter'] = checkpoint['iter']
        self.monitor['prev_IoU'] = checkpoint['prev_IoU']
        self.cfg = checkpoint['cfg']
        self.log.info('>> trained parameters are loaded from {%s}' % file_name)
        self.log.info(">> current training status : %.4f IoU" % self.monitor['prev_IoU'])

    def save_trained_network_params(self, e):

        # save trained model
        _ = save_read_latest_checkpoint_num(os.path.join(self.save_dir), e, isSave=True)
        file_name = self.save_dir + '/saved_chk_point_%d.pt' % e
        check_point = {
            'epoch': e,
            'model_state_dict': self.model.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'opt': self.opt.state_dict(),
            'prev_IoU': self.monitor['prev_IoU'],
            'iter': self.monitor['iter'],
            'cfg': self.cfg}
        torch.save(check_point, file_name)
        self.log.info(">> current network is saved ...")
        remove_past_checkpoint(os.path.join('./', self.save_dir), max_num=self.args.max_num_chkpts)

    def print_status(self, e, tl):
        if (self.rank==0):
            bce = self.monitor['bce']
            cur_lr = self.opt.param_groups[0]['lr']
            self.log.info(f'[E {e:d}, {tl:.2f} hrs left] bce : {bce:.4f} (cur lr: {cur_lr:.7f})')

    def print_training_progress(self, e, b, time):
        if (self.rank == 0):
            if (b >= self.num_batches - 2): sys.stdout.write('\r')
            else: sys.stdout.write('\r [Epoch %02d] %d / %d (%.4f sec/sample), total loss : %.4f'
                                   % (e, b, self.num_batches, time, self.monitor['total_loss'])),
            sys.stdout.flush()

    def print_validation_progress(self, b, num_batchs):
        if (self.rank == 0):
            if (b >= num_batchs - 2): sys.stdout.write('\r')
            else: sys.stdout.write('\r >> validation process (%d / %d) ' % (b, num_batchs)),
            sys.stdout.flush()

    # ------------------------
    # Training
    # ------------------------

    def reform_batch(self, batch, target_index, isTrain=True):
        '''
        Items in batch have sequential data. This function selects one from the sequence.
        '''

        target_batch = {}
        for key, value in batch.items():
            if (isTrain): target_batch[key] = value[:, target_index]
            else: target_batch[key] = value[target_index].unsqueeze(0)
        return target_batch

    def train(self, batch):

        self.opt.zero_grad()
        if (self.cfg['bool_mixed_precision']):

            with torch.cuda.amp.autocast():
                # prediction
                pred = self.model(batch, self.dtype, self.rank)

                # loss calculation
                target_batch = self.reform_batch(batch, target_index=-1, isTrain=True)
                loss = self.loss(pred, target_batch)

            # back-propagation
            self.scaler.scale(loss).backward()
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            # prediction
            pred = self.model(batch, self.dtype, self.rank)

            # loss calculation
            loss = torch.zeros(1).to(pred[self.cfg['target'][0]][0])
            target_batch = self.reform_batch(batch, target_index=-1, isTrain=True)

            # main loss
            weights = self.cfg['training_params']['weights']
            losses = self.loss.main(pred, target_batch)
            for key, item in losses.items():
                loss += weights[key] * item['loss']

            # intermediate loss
            losses = self.loss.intermediate(pred, target_batch)
            for key, item in losses.items():
                loss += weights['intm'] * weights[key] * item['loss']

            # offset l1 loss
            if (pred['offsets'] is not None):
                losses = self.loss.offset(pred, target_batch)
                for key, item in losses.items():
                    loss += weights['offset'] * weights[key] * item['loss']

            # back-propagation
            loss.backward()
            if self.args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.opt.step()


        self.monitor['bce'] += loss.item()
        self.monitor['total_loss'] = loss.item()

        # increase iteration number
        self.monitor['iter'] += 1


    # ------------------------
    # Validation
    # ------------------------
    def eval(self, dataset, dataloader, sampler, e):

        num_batch = dataset.num_scenes / (self.args.batch_size * self.world_size)

        # create empty metric
        metrics = {}
        for _, key in enumerate(self.cfg['target']):
            min_vis = 2 if (key in ['vehicle', 'pedestrian']) else None
            metrics[key] = IoUMetric(label_indices=self.cfg['label_indices'][key],
                                     min_visibility=min_vis,
                                     target_class=key)


        # set to evaluation mode
        self.mode_selection(isTrain=False)
        with torch.no_grad():
            for b, batch in enumerate(dataloader):
                target_batch = self.reform_batch(batch, target_index=-1, isTrain=True)
                pred = self.model(batch, self.dtype, rank=self.rank)
                for key, item in metrics.items():
                    item.update(pred[key][0], target_batch)
                self.print_validation_progress(b, num_batch-1)

        IoU = {}
        for key, item in metrics.items():
            IoU[key] = item.compute()['@0.50']
        IoU_mean = np.mean([item for _, item in IoU.items()])

        if (self.rank == 0):
            self.log.info(f">> evaluation results are created .. IoU@0.5: {IoU_mean:.4f}")
            if (self.monitor['prev_IoU'] < IoU_mean):
                self.monitor['prev_IoU'] = IoU_mean
                self.save_trained_network_params(e)