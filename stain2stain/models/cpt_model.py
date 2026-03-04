"""CPT model for inference only (test mode)."""
import os
import torch

from .base_model import BaseModel
from . import networks
import util.util as util


class CPTModel(BaseModel):
    """Contrastive Paired Translation (CPT) - inference only."""

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')
        parser.add_argument('--lambda_GAN', type=float, default=1.0)
        parser.add_argument('--lambda_NCE', type=float, default=1.0)
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'])
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07)
        parser.add_argument('--num_patches', type=int, default=256)
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False)
        parser.set_defaults(pool_size=0)

        parser.add_argument('--lambda_gp', type=float, default=1.0)
        parser.add_argument('--gp_weights', type=str, default='uniform')
        parser.add_argument('--lambda_asp', type=float, default=0.0)
        parser.add_argument('--asp_loss_mode', type=str, default='none')
        parser.add_argument('--n_downsampling', type=int, default=2)
        parser.add_argument('--use_simsiam', type=util.str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--lambda_style', type=float, default=100.0)
        parser.add_argument('--lambda_content', type=float, default=1.0)
        parser.add_argument('--use_styleloss_slicedwasserstein', type=util.str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--use_clsA', type=float, default=0.0)
        parser.add_argument('--use_clsB', type=float, default=0.0)
        parser.add_argument('--use_clsfB', type=float, default=0.0)
        parser.add_argument('--lambda_cls', type=float, default=0.0)
        parser.add_argument('--cls_content', type=util.str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--no_clsfB_flow', type=util.str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--lambda_discls', type=float, default=0.1)

        opt, _ = parser.parse_known_args()
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=False,
                n_epochs=20, n_epochs_decay=10
            )
        else:
            raise ValueError(opt.CUT_mode)
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = []
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]
        self.model_names = ['G']

        self.netG = networks.define_G(
            opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG,
            not opt.no_dropout, opt.init_type, opt.init_gain,
            opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt
        )

    def data_dependent_initialize(self, data):
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        if 'label' in input:
            self.labels = torch.tensor(input['label'], dtype=torch.float32).to(self.device)
            if len(self.labels.shape) == 0:
                self.labels = self.labels.unsqueeze(0)
        else:
            self.labels = None
        if 'current_epoch' in input:
            self.current_epoch = input['current_epoch']
        if 'current_iter' in input:
            self.current_iter = input['current_iter']

    def forward(self):
        self.real = self.real_A
        self.fake = self.netG(self.real, layers=[])
        self.fake_B = self.fake[:self.real_A.size(0)]

    def optimize_parameters(self):
        """No-op for inference-only model."""
        pass

    def test(self):
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        pass

    def parallelize(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                setattr(self, 'net' + name, torch.nn.DataParallel(net, self.opt.gpu_ids))

    def print_networks(self, verbose):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = sum(p.numel() for p in net.parameters())
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def load_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_dir = self.save_dir
                load_path = os.path.join(load_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata
                net.load_state_dict(state_dict)
