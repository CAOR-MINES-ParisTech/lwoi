import argparse
import os
import progressbar
import torch
import pyro
import pyro.contrib.gp as gp
import numpy as np
from liegroups.torch import SE3, SO3
import pickle
from dataset import NCLTDataset, KAISTDataset
from filter import KAISTFilter, NCLTFilter
from plots import plot_animation, plot_and_save_traj, plot_and_save_cate
from scipy.signal import savgol_filter
from train import train_gp, GpOdoFog, GpImu, FNET, HNET

def read_data_nclt(args):
	def set_path_nclt(args, dataset):
		path_odo = os.path.join(args.path_data_base, 'sensor_data', dataset, "wheels.csv")
		path_fog = os.path.join(args.path_data_base, 'sensor_data', dataset, "kvh.csv")
		path_imu = os.path.join(args.path_data_base, 'sensor_data', dataset, "ms25.csv")
		path_gt = os.path.join(args.path_data_base, 'ground_truth', "groundtruth_" + dataset + ".csv")
		return path_odo, path_fog, path_imu, path_gt

	def gt2chi(x):
		"""Convert ground truth (position, Euler angle) to SE(3) pose"""
		X = torch.eye(4)
		X[:3, :3] = SO3.from_rpy(x[3:]).as_matrix()
		X[:3, 3] = x[:3]
		return X

	time_factor = 1e6 # ms -> s
	g = torch.Tensor([0, 0, 9.81]) # gravity vector

	def interp_data(x, t, t0):
		x_int = np.zeros((t.shape[0], x.shape[1]))
		x_int[:, 0] = t
		for i in range(1, x.shape[1]):
				x_int[:, i] = np.interp(t, (x[:, 0] - t0) / time_factor, x[:, i])
		return x_int

	datasets = os.listdir(os.path.join(args.path_data_base, 'sensor_data'))
	k = int(args.Delta_t/args.delta_t)
	bar_dataset = progressbar.ProgressBar(max_value=len(datasets))
	for idx_i, dataset_i in enumerate(datasets):
		print("\nDataset name: " + dataset_i)
		path_odo, path_fog, path_imu, path_gt = set_path_nclt(args, dataset_i)

		imu = np.genfromtxt(path_imu, delimiter=",", skip_header=1)
		odo = np.genfromtxt(path_odo, delimiter=",", skip_header=1)
		fog = np.genfromtxt(path_fog, delimiter=",", skip_header=1)
		gt = np.genfromtxt(path_gt, delimiter=",", skip_header=1)

		# time synchronization
		t0 = np.max([fog[0, 0], gt[0, 0], odo[0, 0], imu[0, 0]])
		t_end = np.min([fog[-1, 0], gt[-1, 0], odo[-1, 0], imu[-1, 0]])

		# interpolate all data, with particular attention to angles
		t_end = int((t_end-t0)/time_factor)
		t = np.linspace(0, t_end, num=int(t_end/args.delta_t))

		gt_new = np.zeros((t.shape[0], gt.shape[1]))
		fog_new = np.zeros((t.shape[0], 4))
		fog_unwrap = np.unwrap(fog[:, 1])
		gt_t = (gt[:, 0]-t0)/time_factor
		fog_t = (fog[:, 0]-t0)/time_factor
		i_gt = 0
		i_fog = 0
		i_fog_prev = i_fog
		for j in range(t.shape[0]):
			while gt_t[i_gt] < t[j]:
				i_gt += 1
			while fog_t[i_fog] < t[j]:
				i_fog += 1

			if 	np.abs(gt_t[i_gt]-t[j]) <  np.abs(gt_t[i_gt-1]-t[j]):
				gt_new[j, :] = gt[i_gt, :]
			else:
				gt_new[j, :] = gt[i_gt-1, :]

			if 	np.abs(fog_t[i_fog]-t[j]) <  np.abs(fog_t[i_fog+1]-t[j]):
				fog_new[j, 3] = fog_unwrap[i_fog]-fog_unwrap[i_fog_prev]
				i_fog_prev = i_fog
			else:
				fog_new[j, 3] = fog_unwrap[i_fog+1]-fog_unwrap[i_fog_prev]
				i_fog_prev = i_fog+1

		gt_new[:, :4] = interp_data(gt[:, :4], t, t0)
		odo = interp_data(odo, t, t0)
		imu = interp_data(imu, t, t0)

		gt = torch.from_numpy(gt_new[:, 1:])
		imu = torch.from_numpy(imu[:, 1:])
		odo = torch.from_numpy(odo[:, 1:])
		fog = torch.from_numpy(-fog_new[:, 1:])

		# take IMR gyro and accelerometer
		imu = imu[:, [6, 7, 8, 3, 4, 5]]

		error = (fog-fog.float().double()).norm() + (imu-imu.float().double()).norm() + \
		        (odo-odo.float().double()).norm() + (gt-gt.float().double()).norm()
		if error > 0.1:
			print("conversion double -> float error ! ! !")

		fog = fog.float()
		imu = imu.float()
		odo = odo.float()
		gt = gt.float()
		# offset position to 0
		gt[:, :3] = gt[:, :3]-gt[0, :3]

		v_gt = torch.zeros(gt.shape[0], 3)
		for j in range(3):
			p_gt_smooth = torch.from_numpy(savgol_filter(gt[:, j], 5, 2))
			v_gt[1:, j] = (p_gt_smooth[1:]-p_gt_smooth[:-1])/args.delta_t

		N_max = torch.ceil(torch.Tensor([t.shape[0]/k])).int().item()
		chi =  torch.eye(4).repeat(N_max, 1, 1)
		y_odo_fog = torch.eye(4).repeat(N_max, 1, 1)
		u_odo_fog = torch.zeros(N_max, k, 3)
		u_imu = torch.zeros(N_max, k, 6)
		y_imu = torch.zeros(N_max, 9)

		i_odo = 0
		i = 0
		bar_dataset_i = progressbar.ProgressBar(t.shape[0])
		while i_odo + k < t.shape[0]:
			u_odo_fog[i] = torch.cat((odo[i_odo:i_odo+k],
			                          fog[i_odo:i_odo+k, 2].unsqueeze(-1)), 1)
			u_imu[i] = imu[i_odo:i_odo+k]
			chi_end = gt2chi(gt[i_odo+k])
			chi[i] =  gt2chi(gt[i_odo])
			chi_i = chi[i]

			y_odo_fog[i] = SE3.from_matrix(chi_i).inv().dot(SE3.from_matrix(chi_end)).as_matrix()

			v_i = v_gt[i_odo]
			v_end = v_gt[i_odo+k]

			y_imu[i]  =  torch.cat((
			 	SO3.from_matrix(chi_i[:3, :3].t().mm(chi_end[:3, :3])).log(),
				chi_i[:3, :3].t().mv(v_end-v_i-g*args.Delta_t),
				chi_i[:3, :3].t().mv(chi_end[:3, 3]-chi_i[:3, 3]-v_i*args.Delta_t-1/2*g*args.Delta_t**2)
			), 0)

			i_odo += k
			i += 1
			if i_odo % 100 == 0:
				bar_dataset_i.update(i_odo)

		mondict = {'t': t[:i],
				   'chi': chi[:i],
				   'u_imu': u_imu[:i],
				   'u_odo_fog': u_odo_fog[:i],
				   'y_odo_fog': y_odo_fog[:i],
				   'y_imu': y_imu[:i],
				   'name': dataset_i
				   }
		bar_dataset.update(idx_i)
		print("\nNumber of points: {}".format(i))
		with open(args.path_data_save + dataset_i +".p", "wb") as file_pi:
			pickle.dump(mondict, file_pi)

def read_data_kaist(args):
	def set_path_kaist(args, dataset):
		path_odo = os.path.join(args.path_data_base, dataset, "sensor_data", "encoder.csv")
		path_fog = os.path.join(args.path_data_base, dataset, "sensor_data", "fog.csv")
		path_imu = os.path.join(args.path_data_base, dataset, "sensor_data", "xsens_imu.csv")
		path_gt = os.path.join(args.path_data_base, dataset, "global_pose.csv")
		return path_odo, path_fog, path_imu, path_gt

	def gt2chi(x):
		X = torch.eye(4)
		X[0] = x[:4]
		X[1] = x[4:8]
		X[2] = x[8:12]
		X[:3, :3] = SO3.from_matrix(X[:3, :3], normalize=True).as_matrix()
		return X

	time_factor = 1e9 # ns -> s
	g = torch.Tensor([0, 0, -9.81]) # gravity vector
	threshold_odo = 30 # for removing outlier

	def interp_data(x, t, t0):
		x_int = np.zeros((t.shape[0], x.shape[1]))
		for i in range(1, x.shape[1]):
			x_int[:, i] = t if i == 0 else np.interp(t, (x[:, 0] - t0) / time_factor, x[:, i])
		return x_int

	datasets = os.listdir(args.path_data_base)
	k = int(args.Delta_t/args.delta_t)

	bar_dataset = progressbar.ProgressBar(max_value=len(datasets))
	for idx_i, dataset_i in enumerate(datasets):
		print("\nDataset name: " + dataset_i)

		path_odo, path_fog, path_imu, path_gt = set_path_kaist(args, dataset_i)

		imu = np.genfromtxt(path_imu, delimiter=",")
		odo = np.genfromtxt(path_odo, delimiter=",")
		fog = np.genfromtxt(path_fog, delimiter=",")
		gt = np.genfromtxt(path_gt, delimiter=",")

		# Urban00-05 and campus00 have only quaternion and Euler data
		# Must be considered in Dataset class
		imu_present = imu.shape[1] > 10
		if not imu_present:
			imu = np.zeros((odo.shape[0], 17))
			imu[:, 0] = odo[:, 0]
			print("No IMU data for dataset " + dataset_i)

		# time synchronization
		t0 = np.max([fog[0, 0], gt[0, 0], odo[0, 0], imu[0, 0]])
		t_end = np.min([fog[-1, 0], gt[-1, 0], odo[-1, 0], imu[-1, 0]])

		# interpolate all
		# Transform differential measurement into integrated measurement
		t_end = int((t_end-t0)/time_factor)
		t = np.linspace(0, t_end, num=int(t_end/args.delta_t))

		gt_new = np.zeros((t.shape[0], gt.shape[1]))
		fog_new = np.zeros((t.shape[0], 4))
		gt_t = (gt[:, 0]-t0)/time_factor
		fog_t = (fog[:, 0]-t0)/time_factor
		i_gt = 0
		i_fog = 0
		i_fog_prev = i_fog
		for j in range(t.shape[0]):
			while gt_t[i_gt] < t[j]:
				i_gt += 1
			while fog_t[i_fog] < t[j]:
				i_fog += 1

			if 	np.abs(gt_t[i_gt]-t[j]) <  np.abs(gt_t[i_gt-1]-t[j]):
				gt_new[j, :] = gt[i_gt, :]
			else:
				gt_new[j, :] = gt[i_gt-1, :]

			if 	np.abs(fog_t[i_fog]-t[j]) <  np.abs(fog_t[i_fog+1]-t[j]):
				fog_new[j, 1:] = np.sum(fog[i_fog_prev:i_fog, 1:], axis=0)
			else:
				fog_new[j, 1:] = np.sum(fog[i_fog_prev:i_fog+1, 1:], axis=0)
			i_fog_prev = i_fog

		# interpolate position
		gt_new[:, [0, 4, 8, 12]] = interp_data(gt[:, [0, 4, 8, 12]], t, t0)

		gt_new[:, 0] = t
		fog_new[:, 0] = t

		odo = interp_data(odo, t, t0)
		imu = interp_data(imu, t, t0)

		gt = torch.from_numpy(gt_new[:, 1:])
		imu = torch.from_numpy(imu[:, 1:])
		odo = torch.from_numpy(odo[:, 1:])
		fog = torch.from_numpy(fog_new[:, 1:])

		# take IMR gyro and accelerometer
		imu = imu[:, 7:13]

		# Transform integrated measurement into differential measurement
		odo[1:, :] = odo[1:, :] - odo[:-1, :]
		odo[0, :] = 0

		# remove outlier
		diff_odo = (odo[:, 1]-odo[:, 0]).numpy()
		idx_outlier = np.where(np.abs(diff_odo) > threshold_odo)
		print("outliers in odometer: {:.2f}%".format(len(idx_outlier[0])/diff_odo.shape[0]))
		while len(idx_outlier[0]) > 0:
			for idx in idx_outlier[0]:
				diff_odo[idx] = (diff_odo[idx+1]+diff_odo[idx-1])/2
			idx_outlier = np.where(np.abs(diff_odo) > threshold_odo)

		# offset position to 0
		for j in range(3):
			gt[:, 3+4*j] = gt[:, 3+4*j]-gt[0, 3+4*j]

		error = (fog-fog.float().double()).norm() + (imu-imu.float().double()).norm() + \
		        (odo-odo.float().double()).norm() + (gt-gt.float().double()).norm()
		if error > 0.1:
			print("conversion double -> float error ! ! !")

		fog = fog.float()
		imu = imu.float()
		odo = odo.float()
		gt = gt.float()

		v_gt = torch.zeros(gt.shape[0], 3)
		for j in range(3):
			p_gt_smooth = torch.from_numpy(savgol_filter(gt[:, 3+4*j], 5, 2))
			v_gt[1:, j] = (p_gt_smooth[1:]-p_gt_smooth[:-1])/args.delta_t

		# max number of measurements for this dataset
		N_max = torch.ceil(torch.Tensor([t.shape[0]/k])).int().item()
		chi =  torch.eye(4).repeat(N_max, 1, 1)
		y_odo_fog = torch.eye(4).repeat(N_max, 1, 1)
		u_odo_fog = torch.zeros(N_max, k, 3)
		u_imu = torch.zeros(N_max, k, 6)
		y_imu = torch.zeros(N_max, 9)	

		i_odo = 0
		i = 0
		bar_dataset_i = progressbar.ProgressBar(t.shape[0])
		while i_odo + k < t.shape[0]:
			u_odo_fog[i] = torch.cat((odo[i_odo:i_odo+k],
			                          fog[i_odo:i_odo+k, 2].unsqueeze(-1)), 1)
			chi_end = gt2chi(gt[i_odo+k])
			chi[i] = gt2chi(gt[i_odo])
			chi_i = chi[i]

			y_odo_fog[i] = SE3.from_matrix(chi_i).inv().dot(SE3.from_matrix(chi_end)).as_matrix()

			if imu_present:
				u_imu[i] = imu[i_odo:i_odo + k]
				v_i = v_gt[i_odo]
				v_end = v_gt[i_odo+k]

				y_imu[i] =  torch.cat((
				 	SO3.from_matrix(chi_i[:3, :3].t().mm(chi_end[:3, :3])).log(),
					chi_i[:3, :3].t().mv(v_end-v_i- g*args.Delta_t),
					chi_i[:3, :3].t().mv(chi_end[:3, 3]-chi_i[:3, 3]-v_i*args.Delta_t-1/2*g*args.Delta_t**2)
				),0)

			i_odo += k
			i += 1
			if i_odo % 100 == 0:
				bar_dataset_i.update(i_odo)

		mondict = {'t': t[:i],
				   'chi': chi[:i],
				   'u_imu': u_imu[:i],
				   'u_odo_fog': u_odo_fog[:i],
				   'y_odo_fog': y_odo_fog[:i],
				   'y_imu': y_imu[:i],
				   'name': dataset_i
				   }
		bar_dataset.update(idx_i)
		print("\nNumber of points: {}\n".format(i))
		with open(args.path_data_save + dataset_i +".p", "wb") as file_pi:
			pickle.dump(mondict, file_pi)

def set_gp_imu(args, dataset):
	path_gp_imu = args.path_temp + "gp_imu"
	if args.nclt: # this is just for correct dimension
		u, y = dataset.get_train_data(1, gp_name='GpImu')
	else:
		u, y = dataset.get_test_data(1, gp_name='GpImu')

	hnet_dict = torch.load(path_gp_imu + "hnet.p")
	lik_dict = torch.load(path_gp_imu + "likelihood.p")
	kernel_dict = torch.load(path_gp_imu + "kernel.p")
	gp_dict = torch.load(path_gp_imu + "gp_h.p")

	hnet = HNET(args, u.shape[2], args.kernel_dim)
	hnet.load_state_dict(hnet_dict)
	def hnet_fn(x):
		return pyro.module("HNET", hnet)(x)

	Xu = u[torch.arange(0, u.shape[0], step=int(u.shape[0]/args.num_inducing_point)).long()]
	lik_h = gp.likelihoods.Gaussian(name='lik_h', variance=torch.ones(9, 1))
	lik_h.load_state_dict(lik_dict)

	kernel_h = gp.kernels.Matern52(input_dim=args.kernel_dim, lengthscale=torch.ones(args.kernel_dim)).\
		warp(iwarping_fn=hnet_fn)
	kernel_h.load_state_dict(kernel_dict)
	gp_h = gp.models.VariationalSparseGP(u, u.new_ones(9, u.shape[0]), kernel_h, Xu, num_data=dataset.num_data,
										 likelihood=lik_h, mean_function=None, name='GP_h', whiten=True, jitter=1e-4)
	gp_h.load_state_dict(gp_dict)

	gp_imu = GpImu(args, gp_h, dataset)
	gp_imu.normalize_factors = torch.load(path_gp_imu + "normalize_factors.p")
	return gp_imu

def set_gp_odo_fog(args, dataset):
	path_gp_odo_fog = args.path_temp + "gp_odo_fog"
	if args.nclt: # this is just for correct dimension
		u, y = dataset.get_train_data(1, gp_name='GpOdoFog')
	else:
		u, y = dataset.get_test_data(1, gp_name='GpOdoFog')

	fnet_dict = torch.load(path_gp_odo_fog + "fnet.p")
	lik_dict = torch.load(path_gp_odo_fog + "likelihood.p")
	kernel_dict = torch.load(path_gp_odo_fog + "kernel.p")
	gp_dict = torch.load(path_gp_odo_fog + "gp_f.p")

	fnet = FNET(args, u.shape[2], args.kernel_dim)
	fnet.load_state_dict(fnet_dict)
	def fnet_fn(x):
		return pyro.module("FNET", fnet)(x)

	Xu = u[torch.arange(0, u.shape[0], step=int(u.shape[0]/args.num_inducing_point)).long()]
	lik_f = gp.likelihoods.Gaussian(name='lik_f', variance=torch.ones(6, 1))
	lik_f.load_state_dict(lik_dict)

	kernel_f = gp.kernels.Matern52(input_dim=args.kernel_dim, lengthscale=torch.ones(args.kernel_dim)).\
		warp(iwarping_fn=fnet_fn)
	kernel_f.load_state_dict(kernel_dict)
	gp_f = gp.models.VariationalSparseGP(u, u.new_ones(6, u.shape[0]), kernel_f, Xu, num_data=dataset.num_data,
										 likelihood=lik_f, mean_function=None, name='GP_f', whiten=True, jitter=1e-4)
	gp_f.load_state_dict(gp_dict)

	gp_odo_fog = GpOdoFog(args, gp_f, dataset)
	gp_odo_fog.normalize_factors = torch.load(path_gp_odo_fog + "normalize_factors.p")
	return gp_odo_fog

def post_tests(args, dataset, filter_original):
	gp_odo_fog = set_gp_odo_fog(args, dataset)
	gp_imu = set_gp_imu(args, dataset)
	filter_corrected = args.filter(args, dataset, gp_odo_fog=gp_odo_fog, gp_imu=gp_imu)
	bar_dataset = progressbar.ProgressBar(max_value=len(dataset.datasets))

	for i in range(len(dataset.datasets)):
		dataset_name = dataset.datasets[i]
		if dataset_name in dataset.datasets_test:
			type_dataset = ", Test dataset"
		elif dataset_name in dataset.datasets_validation:
			type_dataset = ", Cross-validation dataset"
		else:
			type_dataset = ", Training dataset"

		t, x0, u_odo_fog, y_imu = dataset.get_filter_data(dataset_name)
		P0 = torch.zeros(15, 15)
		u_odo = u_odo_fog[..., :2]
		u_fog = u_odo_fog[..., 2:]

		x_corrected, P_corrected = filter_corrected.run(t, x0, P0, u_fog, u_odo, y_imu, args.compare)
		x_original, P_original = filter_original.run(t, x0, P0, u_fog, u_odo, y_imu, args.compare)

		_, chi = dataset.get_ground_truth_data(dataset_name)
		t = np.linspace(0, args.Delta_t*t.shape[0], t.shape[0])

		error_corrected = filter_corrected.compute_error(t, x_corrected, chi, dataset_name)
		error_original = filter_original.compute_error(t, x_original, chi, dataset_name)

		print("\n" + dataset_name + type_dataset + ", dataset size: {}".format(chi.shape[0]))
		print("m-ATE Translation corrected " + args.compare + ": {:.2f} (m-ATE un-corrected ".format(
			error_corrected['mate translation']) + args.compare + ": {:.2f})".format(error_original['mate translation']))
		print("m-ATE Rotation corrected " + args.compare + " : {:.2f} (m-ATE un-corrected ".format(
			error_corrected['mate rotation']*180/np.pi) + args.compare + ": {:.2f})".format(error_original['mate rotation']*180/np.pi))
		bar_dataset.update(i)


def launch(args):
	if args.nclt:
		args.filter = NCLTFilter
		args.dataset_name = "nclt"
		args.cross_validation_sequences = ['2012-10-28', '2012-11-04', '2012-11-16', '2012-11-17']
		args.test_sequences = ['2012-12-01', '2013-01-10', '2013-02-23', '2013-04-05']
	else:
		args.filter = KAISTFilter
		args.dataset_name = "Kaist"
		args.cross_validation_sequences = ['urban14', 'urban17']
		args.test_sequences = ['urban15', 'urban16']

	### What to do
	args.read_data = True
	args.train_gp_odo_fog = True
	args.train_gp_imu = True
	args.post_tests = True

	# extract data
	if args.read_data:
		read_data_nclt(args) if args.nclt else read_data_kaist(args)

	dataset = NCLTDataset(args) if args.nclt else KAISTDataset(args)
	filter_original = args.filter(args, dataset)

	# train propagation Gaussian process
	if args.train_gp_odo_fog:
		train_gp(args, dataset, GpOdoFog)

	# train measurement Gaussian process
	if args.train_gp_imu:
		train_gp(args, dataset, GpImu)

	# run models and filters on validation data
	if args.post_tests:
		post_tests(args, dataset, filter_original)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='GP Kaist')
	parser.add_argument('--nclt', type=bool, default=False)

	# paths
	parser.add_argument('--path_data_base', type=str, default="/media/mines/DATA/KAIST/data/")
	parser.add_argument('--path_data_save', type=str, default="data/kaist/")
	parser.add_argument('--path_results', type=str, default="results/kaist/")
	parser.add_argument('--path_temp', type=str, default="temp/kaist/")

	# data extraction
	parser.add_argument('--y_diff_odo_fog_threshold', type=float, default=0.25)
	parser.add_argument('--y_diff_imu_threshold', type=float, default=0.25)

	# model parameters
	parser.add_argument('--delta_t', type=float, default=0.01)
	parser.add_argument('--Delta_t', type=float, default=1)
	parser.add_argument('--num_inducing_point', type=int, default=100)
	parser.add_argument('--kernel_dim', type=int, default=20)

	# optimizer parameters
	parser.add_argument('--epochs', type=int, default=100)
	parser.add_argument('--lr', type=float, default=0.01)
	parser.add_argument('--lr_decay', type=float, default=0.999)
	parser.add_argument('--compare', type=str, default="model")

	args = parser.parse_args()
	launch(args)
