import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.contrib.gp as gp
import pyro.infer as infer
import pyro.optim as optim
from pyro.contrib.gp.util import Parameterized
import numpy as np
from liegroups.torch import SE3, SO3
from utils import jacobian, MultiVariateGaussian
pyro.enable_validation(True)

# batch matrix vector multiplication
bmv = lambda bM, bv: bM.matmul(bv.unsqueeze(-1)).squeeze()

class FNET(nn.Module):
	def __init__(self, args, u_dim, kernel_dim):
		super(FNET, self).__init__()
		self.conv1 = nn.Conv1d(u_dim, 100, u_dim, stride=5)
		self.conv2 = nn.Conv1d(100, 50, u_dim, stride=5)
		self.fc = nn.Linear(200, kernel_dim)

	def forward(self, x):
		x = F.relu(self.conv2(F.relu(self.conv1(torch.transpose(x, -1, -2)))))
		x = self.fc(x.view(x.shape[0], -1))
		return x

class HNET(FNET):
	def __init__(self, args, u_dim, kernel_dim):
		super(HNET, self).__init__(args, u_dim, kernel_dim)
		self.conv1 = nn.Conv1d(u_dim, 100, u_dim, stride=4)
		self.conv2 = nn.Conv1d(100, 50, u_dim, stride=4)
		self.fc = nn.Linear(250, kernel_dim)

class GpOdoFog(Parameterized):
	name = 'GpOdoFog'
	def __init__(self, args, gp_f, dataset):
		super(GpOdoFog, self).__init__(name='GpOdoFog')
		self.gp_f = gp_f
		self.normalize_factors = dataset.normalize_factors
		self.calibration_parameters = dataset.calibration_parameters
		self.Delta_t = args.Delta_t
		self.y_diff_threshold = args.y_diff_odo_fog_threshold
		self.nclt = args.nclt

	def unnormalize(self, x_normalized, var="u_odo_fog"):
		x_loc = self.normalize_factors[var + "_loc"].expand_as(x_normalized)
		x_std = self.normalize_factors[var + "_std"].expand_as(x_normalized)
		return x_normalized*x_std  + x_loc # x

	def normalize(self, x, var="u_odo_fog"):
		x_loc = self.normalize_factors[var + "_loc"].expand_as(x)
		x_std = self.normalize_factors[var + "_std"].expand_as(x)
		return (x-x_loc)/x_std # x_normalized

	def model(self):
		pyro.module("GpOdoFog", self)
		return self.gp_f.model()

	def set_data(self, u, y):
		y_pred = self.f_hat(self.unnormalize(u))
		y_diff = self.box_minus(y, y_pred)
		# remove outlier
		idx = (y_diff**2).mean(dim=1).sqrt() < self.y_diff_threshold
		y_diff_normalized =  self.normalize(y_diff, var="y_odo_fog")
		self.gp_f.set_data(u[idx], y_diff_normalized[idx].t())

	def guide(self):
		pyro.module("GpOdoFog", self)
		return self.gp_f.guide()

	def correct(self, x, u_odo, u_fog, compute_G=False, full_cov=False):
		u_odo_fog = torch.cat((u_odo, u_fog), 1).unsqueeze(0)
		u_odo_fog.requires_grad = True
		Xnew = self.normalize(u_odo_fog)

		# take mean to speed up correction
		y_cor_nor, _ = self.gp_f.forward(Xnew, full_cov)

		# # sample corrections and take mean
		# N = 100
		# mean, cov = self.gp_f.forward(Xnew, full_cov=True)
		# y_cor_nor = torch.zeros(6)
		# dist = torch.distributions.MultivariateNormal(loc=mean, cov)
		# for i in range(N):
		# 	y_cor_nor += 1/N * dist.sample()

		y_cor = self.unnormalize(y_cor_nor.t(), var="y_odo_fog").squeeze()
		G_cor = self.correct_cov(u_odo_fog, y_cor, compute_G)
		u_odo_fog.requires_grad = False
		y_cor = y_cor.detach()
		y_cor[[3,4]] = 0 # pitch and roll corrections are set to 0
		G_cor[[3,4], :] = 0
		Rot = SO3.from_rpy(x[3:6]).as_matrix()
		# correct state
		dRot_cor = SO3.exp(y_cor[3:]).as_matrix()
		x[:3] = x[:3] + Rot.mv(SE3.exp(y_cor).as_matrix()[:3, 3])
		x[3:6] = SO3.from_matrix(Rot.mm(dRot_cor)).to_rpy()
		return x, G_cor

	def correct_cov(self, u_odo_fog, x_cor, compute_G):
		G = torch.zeros(u_odo_fog.shape[1], 15, 9)
		if compute_G:
			for i in range(u_odo_fog.shape[1]):
				G[i, :6, :u_odo_fog.shape[2]] = jacobian(u_odo_fog[0, i], x_cor)
		return G

	def forward(self, Xnew, full_cov=False):
		return self.gp_f.forward(Xnew, full_cov)

	def f_hat(self, u):
		u_odo = u[..., :2]
		u_fog = u[..., 2:]
		delta_t = self.Delta_t / u_odo.shape[1]
		Rot_prev = torch.eye(3).repeat(u_odo.shape[0], 1, 1)
		p_prev = torch.zeros(3).repeat(u_odo.shape[0], 1)

		for i in range(u_odo.shape[1]):
			dRot, dp = self.integrate_odo_fog(u_odo[:, i], u_fog[:, i], delta_t)
			Rot = Rot_prev.matmul(dRot)
			p = p_prev + bmv(Rot_prev, dp)
			Rot_prev = SO3.from_matrix(Rot, True).as_matrix()
			p_prev = p
		chi = torch.eye(4).repeat(u_odo.shape[0], 1, 1)
		chi[:, :3, :3] = Rot
		chi[:, :3, 3] = p
		return chi

	def integrate_odo_fog(self, u_odo, u_fog, delta_t):
		if self.nclt:
			v = 1/2*(u_odo[:, 0] + u_odo[:, 1])
		else:
			v, _ = self.encoder2speed(u_odo, delta_t)
		xi = u_odo.new_zeros(u_odo.shape[0], 6)
		xi[:, 0] = v*delta_t
		xi[:, 5] = u_fog.squeeze()
		Rot = SO3.from_rpy(xi[:, 3:]).as_matrix()
		p = xi[:, :3]
		return Rot, p

	def encoder2speed(self, u_odo, delta_t):
		res = self.calibration_parameters["Encoder resolution"]
		r_l = self.calibration_parameters["Encoder left wheel diameter"]
		r_r = self.calibration_parameters["Encoder right wheel diameter"]
		a = self.calibration_parameters["Encoder wheel base"]
		d_l = np.pi * r_l * u_odo[:, 0] / res
		d_r = np.pi * r_r * u_odo[:, 1] / res
		lin_speed = (d_l + d_r) / 2
		ang_speed = (d_l - d_r) / a
		return lin_speed/delta_t, ang_speed/delta_t

	def box_minus(self, chi_1, chi_2):
		return SE3.from_matrix(chi_2).inv().dot(SE3.from_matrix(chi_1)).log()

class GpImu(Parameterized):
	name = 'GpImu'
	def __init__(self, args, gp_h, dataset):
		super(GpImu, self).__init__(name='GpImu')
		self.gp_h = gp_h
		self.normalize_factors = dataset.normalize_factors
		self.delta_t = args.delta_t
		self.y_diff_threshold = args.y_diff_imu_threshold

	def unnormalize(self, x_normalized, var="u_imu"):
		x_loc = self.normalize_factors[var + "_loc"].expand_as(x_normalized)
		x_std = self.normalize_factors[var + "_std"].expand_as(x_normalized)
		x = x_normalized*x_std  + x_loc
		return x

	def normalize(self, x, var="u_imu"):
		x_loc = self.normalize_factors[var + "_loc"].expand_as(x)
		x_std = self.normalize_factors[var + "_std"].expand_as(x)
		x_normalized = (x-x_loc)/x_std
		return x_normalized

	def model(self):
		pyro.module("GpImu", self)
		return self.gp_h.model()

	def set_data(self, u, y):
		y_pred = self.h_hat(self.unnormalize(u))
		y_diff = y - y_pred
		# remove outlier
		idx = (y_diff**2).mean(dim=1).sqrt() < self.y_diff_threshold
		y_diff_normalized =  self.normalize(y_diff, var="y_imu")
		self.gp_h.set_data(u[idx], y_diff_normalized[idx].t())

	def guide(self):
		pyro.module("GpImu", self)
		return self.gp_h.guide()

	def correct(self, u_imu, full_cov=False):
		u_imu.requires_grad = True
		Xnew = self.normalize(u_imu.unsqueeze(0))
		y_cor_nor, _ = self.forward(Xnew, full_cov)
		y_cor = self.unnormalize(y_cor_nor.t(), var="y_imu").squeeze()
		J_cor = self.correct_cov(u_imu, y_cor)
		y_cor = y_cor.detach()
		y_cor[[1, 2]] = 0 # pitch and roll corrections are set to 0
		J_cor[[1, 2], :] = 0
		u_imu.requires_grad = False
		return y_cor, J_cor

	def correct_cov(self, u_imu, y_cor):
		J = torch.zeros(u_imu.shape[0], 9, 6)
		for i in range(u_imu.shape[0]):
			J[i] = jacobian(u_imu[i], y_cor)
		return J

	def forward(self, Xnew, full_cov=False):
		return self.gp_h.forward(Xnew, full_cov)

	def h_hat(self, u):
		delta_R_prev = torch.eye(3).repeat(u.shape[0], 1, 1)
		delta_v_prev = torch.zeros(3).repeat(u.shape[0], 1)
		delta_p_prev = torch.zeros(3).repeat(u.shape[0], 1)
		for k in range(u.shape[1]):
			delta_R = delta_R_prev.matmul(SO3.exp(u[:, k, :3]*self.delta_t).as_matrix())
			delta_v = delta_v_prev + bmv(delta_R, u[:, k, 3:])*self.delta_t
			delta_p = delta_p_prev + delta_v*self.delta_t + bmv(delta_R, u[:, k, 3:]*self.delta_t)*(self.delta_t**2)/2
			delta_R_prev = SO3.from_matrix(delta_R, normalize=True).as_matrix()
			delta_v_prev = delta_v
			delta_p_prev = delta_p

		return torch.cat((SO3.from_matrix(delta_R).log(),
		               delta_v,
		               delta_p), 1)

def preprocessing(args, dataset, gp):
	# compute error without correction and factors for normalizing target
	print("Starting preprocessing " + dataset.name + ", " + gp.name)
	validation_length = 0
	test_length = 0
	num_train = 0
	if gp.name == "GpOdoFog":
		y_odo_fog_loc = torch.zeros(6) # mean has to be set to zero
		def get_error(i, type_dataset='train'):
			if type_dataset == 'train':
				u, y = dataset.get_train_data(i, gp.name)
			elif type_dataset == 'validation':
				u, y = dataset.get_validation_data(i, gp.name)
			else:
				u, y = dataset.get_test_data(i, gp.name)
			u_unnormalize = gp.unnormalize(u)
			y_hat = gp.f_hat(u_unnormalize)
			y_diff = gp.box_minus(y, y_hat)
			return y_diff[(y_diff**2).mean(dim=1).sqrt() < args.y_diff_odo_fog_threshold]

		y_odo_fog_std = torch.zeros(0, y_odo_fog_loc.shape[0])
		for i in range(len(dataset.datasets_train)):
			y_diff = get_error(i, 'train')
			y_odo_fog_std = torch.cat((y_odo_fog_std, (y_diff)**2), 0)
			num_train += y_diff.shape[0]

		mate_translation = 0
		mate_rotation = 0
		for i in range(len(dataset.datasets_validation)):
			y_diff = get_error(i, 'validation')
			mate_translation += y_diff[:, :3].abs().sum()
			mate_rotation += y_diff[:, 3:].abs().sum()
			validation_length += y_diff.shape[0]
		mate_translation = mate_translation/validation_length
		mate_rotation = mate_rotation/validation_length
		mate_validation = {'mate_translation': mate_translation,
		                   'mate_rotation':  mate_rotation}

		mate_translation = 0
		mate_rotation = 0
		for i in range(len(dataset.datasets_test)):
			y_diff = get_error(i, 'test')
			mate_translation += y_diff[:, :3].abs().sum()
			mate_rotation += y_diff[:, 3:].abs().sum()
			test_length += y_diff.shape[0]
		mate_translation = mate_translation/test_length
		mate_rotation = mate_rotation/test_length
		mate_test = {'mate_translation': mate_translation,
		                   'mate_rotation':  mate_rotation}

		y_odo_fog_std = y_odo_fog_std.mean(dim=0).sqrt()
		y_odo_fog_std[y_odo_fog_std == 0] = 1
		gp.normalize_factors['y_odo_fog_loc'] = y_odo_fog_loc
		gp.normalize_factors['y_odo_fog_std'] = y_odo_fog_std
		mate = {'validation': mate_validation,
		        'test': mate_test}
		print("Number of training points: " + str(num_train))
		print("Number of evaluation points: " + str(test_length))
		print("End of preprocessing " + dataset.name + ", " + gp.name)
		return mate
	else:
		y_imu_loc = torch.zeros(9) # mean has to be set to zero

		def get_error(i, type_dataset='train'):
			if type_dataset == 'train':
				u, y = dataset.get_train_data(i, gp.name)
			elif type_dataset == 'validation':
				u, y = dataset.get_validation_data(i, gp.name)
			else:
				u, y = dataset.get_test_data(i, gp.name)
			u_unnormalize = gp.unnormalize(u)
			y_hat = gp.h_hat(u_unnormalize)
			y_diff = y - y_hat
			return y_diff[y_diff.abs().mean(dim=1) < args.y_diff_imu_threshold]

		y_imu_std = torch.zeros(0, y_imu_loc.shape[0])
		for i in range(len(dataset.datasets_train)):
			y_diff = get_error(i, 'train')
			y_imu_std = torch.cat((y_imu_std, (y_diff)**2), 0)
			num_train += y_diff.shape[0]

		rmse_delta_R = 0
		rmse_delta_v = 0
		rmse_delta_p = 0
		for i in range(len(dataset.datasets_validation)):
			y_diff = get_error(i, 'validation')
			rmse_delta_R += (y_diff[:, :3]**2).sum()
			rmse_delta_v += (y_diff[:, 3:6]**2).sum()
			rmse_delta_p += (y_diff[:, 6:9]**2).sum()
			validation_length += y_diff.shape[0]
		rmse_delta_R = (rmse_delta_R/validation_length).sqrt()
		rmse_delta_v = (rmse_delta_v/validation_length).sqrt()
		rmse_delta_p = (rmse_delta_p/validation_length).sqrt()
		rmse_validation = {'rmse_delta_R': rmse_delta_R,
		                   'rmse_delta_v':  rmse_delta_v,
		                   'rmse_delta_p': rmse_delta_p}

		rmse_delta_R = 0
		rmse_delta_v = 0
		rmse_delta_p = 0
		for i in range(len(dataset.datasets_test)):
			y_diff = get_error(i, 'test')
			rmse_delta_R += (y_diff[:, :3]**2).sum()
			rmse_delta_v += (y_diff[:, 3:6]**2).sum()
			rmse_delta_p += (y_diff[:, 6:9]**2).sum()
			test_length += y_diff.shape[0]
		rmse_delta_R = (rmse_delta_R/test_length).sqrt()
		rmse_delta_v = (rmse_delta_v/test_length).sqrt()
		rmse_delta_p = (rmse_delta_p/test_length).sqrt()
		rmse_test = {'rmse_delta_R': rmse_delta_R,
		                   'rmse_delta_v':  rmse_delta_v,
		                   'rmse_delta_p': rmse_delta_p}

		y_imu_std = y_imu_std.mean(dim=0).sqrt()
		y_imu_std[y_imu_std == 0] = 1
		gp.normalize_factors['y_imu_loc'] = y_imu_loc
		gp.normalize_factors['y_imu_std'] = y_imu_std
		rmse = {'validation': rmse_validation,
		        'test': rmse_test}
		print("Number of training points: " + str(num_train))
		print("Number of cross-validation points: " + str(validation_length))
		print("Number of test points: " + str(test_length))
		print("End of preprocessing " + dataset.name + ", " + gp.name)
		return rmse

def train_loop(dataset, gp, svi, epoch):
	if epoch == 1:
		u, y = dataset.get_train_data(0, gp.name)
		for i in range(1, len(dataset.datasets_train)):
			u_i, y_i = dataset.get_train_data(i, gp.name)
			u = torch.cat((u, u_i), 0)
			y = torch.cat((y, y_i), 0)
		u, y = specific_to_kaist_imu(dataset, gp, u, y)
		gp.set_data(u, y)
	loss = svi.step()
	print('Train Epoch: {:2d} \tLoss: {:.6f}'.format(epoch, loss))

def specific_to_kaist_imu(dataset, gp, u, y):
	if dataset.name == "Kaist" and gp.name == "GpImu":
		print("Removing points without IMU")
		# remove input for sequence without imu
		u_true = np.ones(u.shape[0])
		for i in range(u.shape[0]):
			if u[i].sum() < 1e-5:
				u_true[i] = 0
		u = u[u_true]
		y = y[u_true]
	return u, y



def save_gp(args, gp_model, kernel_net):
	kernel_net.eval()
	gp_model.eval()
	if gp_model.name == 'GpOdoFog':
		name = "gp_odo_fog"
		torch.save(gp_model.gp_f.state_dict(), args.path_temp + name + "gp_f.p")
		torch.save(gp_model.gp_f.kernel.state_dict(), args.path_temp + name + "kernel.p")
		torch.save(gp_model.gp_f.likelihood.state_dict(), args.path_temp + name + "likelihood.p")
		torch.save(kernel_net.state_dict(), args.path_temp + name + "fnet.p")
	else:
		name = 'gp_imu'
		torch.save(gp_model.gp_h.state_dict(), args.path_temp + name + "gp_h.p")
		torch.save(gp_model.gp_h.kernel.state_dict(), args.path_temp + name + "kernel.p")
		torch.save(gp_model.gp_h.likelihood.state_dict(), args.path_temp + name + "likelihood.p")
		torch.save(kernel_net.state_dict(), args.path_temp + name + "hnet.p")

	torch.save(gp_model.normalize_factors, args.path_temp + name + "normalize_factors.p")

	print(gp_model.name + " saved")


def train_gp(args, dataset, gp_class):
	u, y = dataset.get_train_data(0, gp_class.name)  if args.nclt else dataset.get_test_data(1, gp_class.name) # this is only to have a correct dimension

	if gp_class.name == 'GpOdoFog':
		fnet = FNET(args, u.shape[2], args.kernel_dim)
		def fnet_fn(x):
			return pyro.module("FNET", fnet)(x)

		lik = gp.likelihoods.Gaussian(name='lik_f', variance=0.1*torch.ones(6, 1))
		# lik = MultiVariateGaussian(name='lik_f', dim=6) # if lower_triangular_constraint is implemented
		kernel = gp.kernels.Matern52(input_dim=args.kernel_dim,
		                               lengthscale=torch.ones(args.kernel_dim)).warp(iwarping_fn=fnet_fn)
		Xu = u[torch.arange(0, u.shape[0], step=int(u.shape[0]/args.num_inducing_point)).long()]
		gp_model = gp.models.VariationalSparseGP(u, torch.zeros(6, u.shape[0]), kernel, Xu,
		                                     num_data=dataset.num_data, likelihood=lik, mean_function=None,
		                                     name=gp_class.name, whiten=True, jitter=1e-3)
	else:
		hnet = HNET(args, u.shape[2], args.kernel_dim)
		def hnet_fn(x):
			return pyro.module("HNET", hnet)(x)
		lik = gp.likelihoods.Gaussian(name='lik_h', variance=0.1*torch.ones(9, 1))
		# lik = MultiVariateGaussian(name='lik_h', dim=9) # if lower_triangular_constraint is implemented
		kernel = gp.kernels.Matern52(input_dim=args.kernel_dim,
		                               lengthscale=torch.ones(args.kernel_dim)).warp(iwarping_fn=hnet_fn)
		Xu = u[torch.arange(0, u.shape[0], step=int(u.shape[0]/args.num_inducing_point)).long()]
		gp_model = gp.models.VariationalSparseGP(u, torch.zeros(9, u.shape[0]), kernel, Xu,
		                                     num_data=dataset.num_data, likelihood=lik, mean_function=None,
		                                     name=gp_class.name, whiten=True, jitter=1e-4)

	gp_instante = gp_class(args, gp_model, dataset)
	args.mate = preprocessing(args, dataset, gp_instante)

	optimizer = optim.ClippedAdam({"lr": args.lr, "lrd": args.lr_decay})
	svi = infer.SVI(gp_instante.model, gp_instante.guide, optimizer, infer.Trace_ELBO())

	print("Start of training " + dataset.name + ", " + gp_class.name)
	start_time = time.time()
	for epoch in range(1, args.epochs + 1):
		train_loop(dataset, gp_instante, svi, epoch)
		if epoch == 10:
			if gp_class.name == 'GpOdoFog':
				gp_instante.gp_f.jitter = 1e-4
			else:
				gp_instante.gp_h.jitter = 1e-4

	save_gp(args, gp_instante, fnet) if gp_class.name == 'GpOdoFog' else save_gp(args, gp_instante, hnet)
