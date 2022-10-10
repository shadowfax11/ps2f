import torch
import torch.optim
import torch.nn.functional as F
import os, argparse, pdb
import matplotlib.pyplot as plt
import numpy as np
import skimage, skimage.transform
from scipy.io import loadmat, savemat
import dd_models.skip as md
import utils_python.conv_utils as conv
import utils_python.utils_dip as dip

DISP_FREQ = 100
PRNT_FREQ = 20
SAVE_FREQ = 5000

### CUDA setup ###
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using ", device)
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

### Initializations and hyperparameters ### 
PSF_DIR = "psfstacks"
DATA_DIR = "sim_scenes"
SAVE_DIR = "sim_results"

parser = argparse.ArgumentParser()
parser.add_argument('--dip', action='store_true')
parser.add_argument('-d', '--data_fname', type=str, required=True, help="Provide file name for data")
parser.add_argument('-p', '--psf_str', type=str, required=True, help="Provide file name for psfstack")
parser.add_argument('-l1_type', type=str, default='l1', help="Type of L1 regularization")
parser.add_argument('-tv_type', type=str, default='l1', help="Type of TV regularization")
parser.add_argument('-rl1', type=float, default=0.00, help="L1 regularization weight")
parser.add_argument('-rtv', type=float, default=0.00, help="TV regularization weight")
parser.add_argument('-lr', type=float, default=0.01, help="Learning rate")
parser.add_argument('-n', '--niter', type=int, default=10000, help="Optimization algo - no. of iterations")
parser.add_argument('-noise', type=float, default=0.01, help="Noise to be added to the image measurement")
parser.add_argument('-s', '--save_str', type=str, default="temp")
parser.add_argument('--display', action='store_true')
parser.add_argument('--use_3dvol', action='store_true')
args = parser.parse_args()
print(args)

DISP_FLAG = True if args.display else False

### Load PSF ###
ZRANGE  = [-2.5e-3, +2.5e-3]
# ZRANGE = [-3e-6, +3e-6]
DXY     = 6.9e-6
PSF_BG = 0
AX_DOWNSAMPLE = 1
XY_DOWNSAMPLE = 1   # TODO

psf_np = loadmat(os.path.join(PSF_DIR, args.psf_str))
psf_np = psf_np['PSFstack']
D = psf_np.shape[2]
psf_np = (psf_np - PSF_BG).astype(np.float)
if args.use_3dvol:
    psf_np = psf_np/(D*255)
psf_np = psf_np[:,:,::AX_DOWNSAMPLE,:]
Hfor = conv.Convolve3DFFT(psf_np)
D = psf_np.shape[2]
z_vals = np.linspace(ZRANGE[0], ZRANGE[1], num=D)

### Load scene data ###
x_gt = loadmat(os.path.join(DATA_DIR, args.data_fname))
if args.use_3dvol:
    V_gt = x_gt['V_gt']
    V_gt_orig = V_gt
else:
    V_gt_orig = conv.create_3dvol(x_gt['scene_gt'], x_gt['dmap_gt'], z_vals)
    V_gt = conv.create_3dvol(x_gt['scene_gt'], x_gt['dmap_gt'], z_vals, add_poisson_noise=True)
if DISP_FLAG:
    dip.plot_volume(V_gt_orig, True, 'GT Volume')

### Render image ###
NOISE_LVL = args.noise
y_meas_orig = Hfor.forward(conv.np_to_tensor(V_gt_orig.transpose(2,0,1)).type(dtype))
y_meas = Hfor.forward(conv.np_to_tensor(V_gt.transpose(2,0,1)).type(dtype))
y_meas_orig_np = conv.tensor_to_np(y_meas_orig)
y_meas += NOISE_LVL*torch.randn(y_meas.size()).type(dtype)
y_meas[y_meas<0] = 0
y_meas_np = conv.tensor_to_np(y_meas)
C, H, W = y_meas_np.shape
print("Loaded image of size {:2d}x{:4d}x{:4d}".format(C,H,W))
if DISP_FLAG:
    dip.plot(y_meas_np, True, 'Rendered image using given PSF z-stack')
y_meas = torch.nn.Parameter(dip.np_to_tensor(y_meas_np).to(device), requires_grad=False).type(dtype)

### Set up input ###
INPUT_TYPE = 'noise'
PAD = 'reflection'
ACT_FUN = 'LeakyReLU'
REG_NOISE_STD = (0.05)/(10*D)
noise_shape = [D, H, W]
net_input = dip.np_to_tensor(0.0001*np.zeros(noise_shape)).to(device).type(dtype)
net_input.uniform_()
net_input *= 1./(10*D)
if args.dip:
    # pass # TODO
    net_input.detach()
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
else:
    net_input = torch.nn.Parameter(net_input, requires_grad=True)

### Set up network/model ###
def non_neg_constraint(v):
    return F.relu(v)
if args.dip:
    # pass # TODO
    model = md.skip(D, D, 
                    num_channels_down=[32, 32, 32, 32, 32], num_channels_up=[32, 32, 32, 32, 32], 
                    num_channels_skip=[4, 4, 4, 4, 4], upsample_mode='bilinear', downsample_mode='stride',
                    need_sigmoid=True, need_bias=True, pad=PAD, act_fun=ACT_FUN)
    model.to(device).type(dtype)
else: 
    model = dip.NN_constraint()
    model.to(device)

### Set up optimizer and loss functions ###
params = dict()
params['lr'] = args.lr
params['max_iters'] = args.niter
params['lambda_l1'] = args.rl1
params['lambda_tv'] = args.rtv
if args.dip:
    # pass # TODO
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
else:
    optimizer = torch.optim.Adam([net_input], lr=params['lr'])
def loss_fn_lsq(y, y_pred):
    return torch.norm(torch.flatten(y)-torch.flatten(y_pred), p=2)**2
loss_fn = torch.nn.MSELoss(reduction='sum').type(dtype)
# loss_fn = torch.nn.L1Loss(reduction='sum').type(dtype)
# loss_fn = loss_fn_lsq

def run_optimizer(fwd_model, y, model, model_input, optim, loss_fn, params, noise=None, model_input_saved=None):
    """
    fwd_model: function handle for forward model (imaging model). y_est = fwd_model(x_est)
    y: measurements
    model: either a relu function, or deep image prior network
    model_input: self-explanatory
    optim: PyTorch optimizer
    params: training/optimizer hyperparams
    """
    global DISP_FLAG, DISP_FREQ, PRNT_FREQ, SAVE_FREQ, SAVE_DIR, args
    MAX_ITERS = params['max_iters']
    LAMBDA_L1 = params['lambda_l1']
    LAMBDA_TV = params['lambda_tv']
    if args.tv_type=='hessian':
        tvl = dip.TV3DNorm('hessian')
    if args.tv_type=='l1':
        tvl = dip.TV3DNorm('l1')
    loss_iters = []
    iters = []
    for i in range(MAX_ITERS):
        optim.zero_grad()
        if args.dip:
            model_input = model_input_saved + (noise.normal_()*REG_NOISE_STD)
            x_est = model(model_input)
        else:
            x_est = model(model_input)
        y_est = fwd_model(x_est)
        loss_mse = loss_fn(y_est, y)
        if args.l1_type=='l1':
            loss_l1 = LAMBDA_L1*torch.norm(torch.flatten(x_est), p=1)
        if args.l1_type=='l12':
            loss_l1 = LAMBDA_L1*torch.norm(torch.norm(x_est, dim=(2,3)),p=1)
        loss_tv = LAMBDA_TV*tvl(x_est)
        loss = loss_mse + loss_l1 + loss_tv
        loss.backward()
        optim.step()
        loss_i = loss.item()
        loss_iters = np.append(loss_iters, loss_i)
        iters = np.append(iters, i)
        if DISP_FLAG and i%DISP_FREQ==0:
            dip.plot(y_est, False)
        if i%PRNT_FREQ==0 or (i+1)%MAX_ITERS==0:
            print("Epoch: %3d \t Loss: %3.5f \t FIDLoss: %3.5f \t L1Loss: %3.5f \t TVLoss: %3.5f" %(i, loss_i, loss_mse.item(), loss_l1.item(), loss_tv.item()))
        if (i+1)%SAVE_FREQ==0:
            results = {}
            results['x_est'] = dip.tensor_to_np(x_est).transpose(1,2,0)
            results['loss_iters'] = loss_iters
            results['iters'] = iters
            results['params'] = params
            save_str = "{}_{}_{:5d}.mat".format(args.data_fname[:-4], args.save_str, i+1)
            savemat(os.path.join(SAVE_DIR, save_str), results)
    if args.dip:
        model_input = model_input_saved
    return model, model_input, iters, loss_iters

if args.dip:
    model, net_input, iters, loss_iters = run_optimizer(Hfor, y_meas, model, net_input, optimizer, loss_fn, params, noise, net_input_saved)
else:
    model, net_input, iters, loss_iters = run_optimizer(Hfor, y_meas, model, net_input, optimizer, loss_fn, params)
x_est = dip.tensor_to_np(model(net_input)).transpose(1,2,0)
if DISP_FLAG:
    dip.plot_volume(x_est, True, 'Reconstructed Volume')

results = {}
results['y_meas_orig'] = y_meas_orig_np
results['y_meas'] = y_meas_np
results['V_gt'] = V_gt
if not args.use_3dvol:
    results['scene_gt'] = x_gt['scene_gt']
    results['dmap_gt'] = x_gt['dmap_gt']
results['dxy'] = DXY
results['dz'] = z_vals[1]-z_vals[0]
results['x_est'] = x_est
results['loss_iters'] = loss_iters
results['iters'] = iters
results['params'] = params
results['z_vals'] = z_vals
save_str = "{}_{}_{}_{:5d}.mat".format(args.data_fname[:-4], args.noise, args.save_str, args.niter)
savemat(os.path.join(SAVE_DIR, save_str), results)
print("Saved results: {}.mat".format(os.path.join(SAVE_DIR, save_str)))

if args.dip:
    pass # TODO - save model
