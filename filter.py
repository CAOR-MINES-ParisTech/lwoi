import torch
import matplotlib.pyplot as plt
import numpy as np
from liegroups.torch import SE3, SO3, SO2

class NCLTFilter():
	def __init__(self, args, dataset, gp_odo_fog=None, gp_imu=None):
		self.g = -torch.Tensor([0, 0, -9.81])
		self.gp_odo_fog = gp_odo_fog
		self.gp_imu = gp_imu
		self.Delta_t = args.Delta_t
		self.delta_t = args.delta_t
		self.path_results = args.path_results
		self.calibration_parameters = dataset.calibration_parameters

		self.Q = torch.diag(torch.Tensor([
			0.7**2, # sigma_v_r
			0.2**2, # sigma_v_l
			0.001**2, # sigma_v_z
			(0.005/180)**2, # sigma_delta_psi
			(1/180)**2, # sigma_phi
			(1/180)**2, # sigma_theta
			(1/180)**2, # sigma_p
			(1/180)**2, # sigma_q
			(1/180)**2, # sigma_r
			]))

		self.R = torch.diag(torch.Tensor([
			(2/180)**2, # gyro
			(2/180)**2, # gyro
			(2/180)**2, # gyro
			0.1**2, # accelerometer
			0.1**2, # accelerometer
			0.1**2, # accelerometer
			0.7**2, # odo
			0.2**2, # odo
			]))

		# Jacobian, where constant part are pre-computed
		self.F = torch.eye(15, 15)
		# stochastic cloning
		self.F[9:15, 9:15] = 0
		self.F[9:15, :6] = torch.eye(6)

		# noise propagation Jacobian
		self.G = torch.zeros(self.F.shape[0], self.Q.shape[0])
		self.G[2, 2] = torch.eye(1) * args.delta_t # v_z
		self.G[5, 3] = torch.eye(1) * args.delta_t # delta_psi
		self.G[3:5, 4:6] = torch.eye(2) * args.delta_t # phi, theta
		self.G[6:9, 6:9] = torch.eye(3) * args.delta_t # p, q, r

		# state measurement Jacobian
		self.H = torch.zeros(9, self.F.shape[0])
		self.nclt = args.nclt

	def run(self, t, x0, P0, u_fog, u_odo, y_imu, compare):
		N_t = t.shape[0]
		self.x = x0.clone()
		self.P = P0.clone()
		x = torch.zeros(N_t, x0.shape[-1])
		P = torch.zeros(N_t, P0.shape[-1], P0.shape[-1])
		x[0] = self.x
		P[0] = self.P
		delta_t = self.Delta_t/u_odo[0].shape[0]
		for i in range(1, N_t):
			self.propagate(u_odo[i], u_fog[i], delta_t, compare)
			if compare == 'filter':
				self.update(y_imu[i], u_odo[i])
			x[i] = self.x
			P[i] = self.P
		return x, P

	def update(self, u_imu, u_odo):
		y = self.h_imu(u_imu)
		#integrated measurement
		if self.gp_imu:
			y_cor, J_cor = self.gp_imu.correct(u_imu)
			y = y + y_cor
		else:
			J_cor = torch.zeros(u_imu.shape[0], 9, 6)
		K_prefix, S = self.update_cov(J_cor, u_imu)
		self.update_state(K_prefix, S, y-self.h_hat(u_odo))

	def h_imu(self, u):
		"""
		Transforms the imu measurement (gyro, acc) in pre-integrated measurement
		:param u: imu measurements, shape [k, 6]
		:return: pre-integrated measurement
		"""
		delta_R_prev = torch.eye(3)
		delta_v_prev = torch.zeros(3)
		delta_p_prev = torch.zeros(3)
		self.J = torch.zeros(u.shape[0], 9, 8)
		for k in range(u.shape[0]):
			self.J[k, :3, :3] = delta_R_prev*self.delta_t
			self.J[k, 3:6, :3] = -delta_R_prev.mm(self.skew(u[k, 3:]))*self.delta_t
			self.J[k, 3:6, 3:6] = delta_R_prev*self.delta_t
			self.J[k, 3:6, :3] = -1/2*delta_R_prev.mm(self.skew(u[k, 3:]))*(self.delta_t**2)
			self.J[k, 6:9, 3:6] = 1/2*delta_R_prev*(self.delta_t**2)
			delta_R = delta_R_prev.mm(SO3.exp(u[k, :3]*self.delta_t).as_matrix())
			delta_v = delta_v_prev + delta_R.mv(u[k, 3:]*self.delta_t)
			delta_p = delta_p_prev + delta_v*self.delta_t + delta_R.mv(u[k, 3:]*self.delta_t)*(self.delta_t**2)/2
			delta_R_prev = SO3.from_matrix(delta_R, normalize=True).as_matrix()
			delta_v_prev = delta_v
			delta_p_prev = delta_p

		return torch.cat((SO3.from_matrix(delta_R).log(),
					   delta_v,
					   delta_p), 0)

	def h_hat(self, u_odo):
		def odo2speed(u):
			v = 1/2*(u[0]+u[1])
			return torch.Tensor([
					v*torch.cos(self.x_prev[5]),
					v*torch.sin(self.x_prev[5]),
					0])

		# initial speed
		v0 = odo2speed(u_odo[0])

		# end speed
		v_end = odo2speed(u_odo[1])

		R0 = SO3.from_rpy(self.x_prev[3:6]).as_matrix()
		Rend = SO3.from_rpy(self.x[3:6]).as_matrix()

		p0 = self.x_prev[:3]
		p_end = self.x[:3]

		delta_R = SO3.from_matrix(R0.t().mm(Rend)).log()
		delta_v = R0.t().mv(v_end-v0-self.g*self.Delta_t)
		delta_p = R0.t().mv(p_end-p0-v0*self.Delta_t-1/2*self.g*(self.Delta_t**2))

		return torch.cat((delta_R,
		                  delta_v,
		                  delta_p), 0)

	def propagate(self, u_odo, u_fog, delta_t, compare):
		self.x_prev = self.x
		for i in range(u_odo.shape[0]):
			self.integrate_odo_fog(u_odo[i], u_fog[i], delta_t)
		if self.gp_odo_fog:
			self.x, G_cor = self.gp_odo_fog.correct(self.x, u_odo, u_fog, compute_G=(compare == 'filter'))
		else:
			G_cor = torch.zeros(u_odo.shape[0], 15, 9)
		if compare == 'filter':
			self.propagate_cov(u_odo[i], u_fog[i], delta_t, G_cor)
		else:
			self.x[3:6] = SO3.from_rpy(self.x[3:6]).to_rpy()

	def integrate_odo_fog(self, u_odo, u_fog, dt):
		v = 1/2*(u_odo[0]+u_odo[1])
		self.x[0] += v*torch.cos(self.x[5])*dt
		self.x[1] += v*torch.sin(self.x[5])*dt
		self.x[5] += u_fog.squeeze()
		A = torch.zeros(2, 3)
		A[0, 0] = 1
		A[0, 1] = self.x[3].sin() * self.x[4].tan()
		A[0, 2] = self.x[3].cos() * self.x[4].tan()
		A[1, 1] = self.x[3].cos()
		A[1, 2] = -self.x[3].sin()
		self.x[3:5] += A.mv(self.x[6:])*dt

	def propagate_cov(self, u_odo, u_fog, dt, G_cor):
		F = self.F
		G = self.G
		v = 1/2*(u_odo[0]+u_odo[1])
		J = torch.Tensor([[0, 1],[-1, 0]])
		Rot = torch.Tensor([[self.x[5].cos(), self.x[5].sin()], [-self.x[5].sin(), self.x[5].cos()]])
		F[:2, 5] = J.mv(Rot.mv(torch.Tensor([v*dt, 0])))

		A = torch.zeros(2, 3)
		A[0, 0] = 1
		A[0, 1] = self.x[3].sin() * self.x[4].tan()
		A[0, 2] = self.x[3].cos() * self.x[4].tan()
		A[1, 1] = self.x[3].cos()
		A[1, 2] = -self.x[3].sin()
		F[3:5, 6:9] = A*dt
		B = torch.Tensor([[1/2, 1/2], # v_l, v_r to v_forward
		                  [0, 0]])
		F[3, 3] = 1 + self.x[7]*self.x[3].sin()*self.x[4].tan()*dt
		F[3, 4] = (self.x[7]*self.x[3].sin() + self.x[8]*self.x[3].cos())*self.x[4].tan()*dt
		F[4, 3] = self.x[8]*self.x[3].sin()*dt
		G[:2, :2] = SO2.from_angle(self.x[5].unsqueeze(0)).as_matrix().mm(B)*dt

		# add Jacobian correction
		Q = torch.zeros_like(self.P)
		for i in range(G_cor.shape[0]):
			Q += (G+G_cor[i]).mm(self.Q).mm((G+G_cor[i]).t())
		self.P = F.mm(self.P).mm(F.t()) + Q

	def compute_jac_update(self, u_odo):
		J = self.J
		H = self.H

		Rot_prev = SO3.from_rpy(self.x_prev[3:6]).as_matrix()
		Rot_new =  SO3.from_rpy(self.x[3:6]).as_matrix()

		# speed is not in state
		J[0, 3:6, 6:] = -Rot_prev.t()[:3, :2]
		J[-1, 3:6, 6:] = -J[0, 3:6, 6:]
		J[0, 6:9, 6:] = J[0, 6:9, 6:]*self.Delta_t

		v = torch.Tensor([1/2*(u_odo[0][0]+u_odo[0][1]), 0, 0])
		H[:3, 3:6] =  -Rot_prev.t().mm(Rot_new)
		H[:3, 9:12] = -Rot_prev.t()
		H[3:6, 9:12] = -Rot_prev.t().mm(self.skew(self.x[:3]-self.x_prev[:3] - v*self.Delta_t - 1/2*self.g*self.Delta_t**2))
		H[6:9, :3] = Rot_prev.t()
		H[6:9, 12:15] = -Rot_prev.t()
		H[6:9, 9:12] = self.skew(self.x[:3]-self.x_prev[:3] - v*self.Delta_t - 1/2*self.g*self.Delta_t**2)
		return H, J

	def update_cov(self, J_cor, u_odo):
		H, J = self.compute_jac_update(u_odo)
		# add corrected Jacobian on measurement noise covariance

		R = torch.zeros(J.shape[1], J.shape[1])
		for i in range(J_cor.shape[0]):
			J_i = J[i]
			J_i[:, :6] += J_cor[i]
			R += J_i.mm(self.R).mm(J_i.t())

		S = H.mm(self.P).mm(H.t()) + R
		K_prefix = self.P.mm(H.t())
		Id = torch.eye(self.P.shape[-1])
		ImKH = Id - K_prefix.mm(torch.gesv(H, S)[0])
		# *Joseph form* of covariance update for numerical stability.
		self.P = ImKH.mm(self.P).mm(ImKH.transpose(-1, -2)) \
			+ K_prefix.mm(torch.gesv((K_prefix.mm(torch.gesv(R, S)[0])).transpose(-1, -2), S)[0])
		return K_prefix, S

	def update_state(self, K_prefix, S, dy):
		dx = K_prefix.mm(torch.gesv(dy, S)[0]).squeeze(1) # K*dy
		self.x[:3] += dx[:3]
		self.x[3:6] += (SO3.exp(dx[3:6]).dot(SO3.from_rpy(self.x[3:6]))).to_rpy()
		self.x[6:9] += dx[6:9]

	def plot(self, t, x, chi, dataset_name):
		plt.figure()
		plt.xlabel('x (m)')
		plt.ylabel('y (m)')
		p = x[:, :2]
		plt.plot(p[:, 0].numpy(), p[:, 1].numpy())
		plt.plot(chi[:, 0, 3].numpy(), chi[:, 1, 3].numpy())
		plt.title("Trajectory for " + dataset_name)
		plt.grid(True)
		plt.savefig(self.path_results + dataset_name + "plot.png")
		plt.close()


	def compute_cate(self, t, x, chi, dataset_name):
		chi_est = torch.zeros(x.shape[0], 4, 4)
		chi_est[:, :3, :3] = SO3.from_rpy(x[:, 3:6]).as_matrix()
		chi_est[:, :3, 3] = x[:, :3]
		chi_est[:, 3, 3] = 1

		chi_est = SE3.from_matrix(chi_est)
		chi = SE3.from_matrix(chi)
		error = (chi.inv().dot(chi_est)).log()

		cate_translation = error[:, :3].abs().mean(dim=1).cumsum(0)
		cate_rotation = error[:, 3:].abs().mean(dim=1).cumsum(0)

		return cate_translation, cate_rotation


	def compute_mate(self, t, x, chi, dataset_name):
		chi_est = torch.zeros(x.shape[0], 4, 4)
		chi_est[:, :3, :3] = SO3.from_rpy(x[:, 3:6]).as_matrix()
		chi_est[:, :3, 3] = x[:, :3]
		chi_est[:, 3, 3] = 1

		chi_est = SE3.from_matrix(chi_est)
		chi = SE3.from_matrix(chi)
		error = (chi.inv().dot(chi_est)).log()

		mate_translation = error[:, :3].abs().mean()
		mate_rotation = error[:, 3:].abs().mean()
		return mate_translation, mate_rotation

	def compute_error(self, t, x, chi, dataset_name):
		mate_translation, mate_rotation = self.compute_mate(t, x, chi, dataset_name)
		cate_translation, cate_rotation = self.compute_cate(t, x, chi, dataset_name)
		error = {
				 'mate translation': mate_translation,
				 'mate rotation': mate_rotation,
				 'cate translation': cate_translation,
				 'cate rotation': cate_rotation,
				 }
		return error

	def skew(self, x):
		X = torch.zeros(3, 3)
		X[0, 1] = -x[2]
		X[0, 2] = x[1]
		X[1, 0] = x[2]
		X[1, 2] = -x[0]
		X[2, 0] = -x[1]
		X[2, 1] = x[0]
		return X


class KAISTFilter(NCLTFilter):
	def __init__(self, args, dataset, gp_odo_fog=None, gp_imu=None):
		super(KAISTFilter, self).__init__(args, dataset)
		self.gp_odo_fog = gp_odo_fog
		self.gp_imu = gp_imu

	def integrate_odo_fog(self, u_odo, u_fog, dt):
		v, _ = self.encoder2speed(u_odo, dt)
		self.x[0] += v*torch.cos(self.x[5])*dt
		self.x[1] += v*torch.sin(self.x[5])*dt
		self.x[5] += u_fog.squeeze()
		A = torch.zeros(2, 3)
		A[0, 0] = 1
		A[0, 1] = self.x[3].sin() * self.x[4].tan()
		A[0, 2] = self.x[3].cos() * self.x[4].tan()
		A[1, 1] = self.x[3].cos()
		A[1, 2] = -self.x[3].sin()
		self.x[3:5] += A.mv(self.x[6:])*dt

	def propagate_cov(self, u_odo, u_fog, dt, G_cor):
		F = self.F
		G = self.G
		v, _ = self.encoder2speed(u_odo, dt)
		J = torch.Tensor([[0, 1],[-1, 0]])
		Rot = torch.Tensor([[self.x[5].cos(), self.x[5].sin()], [-self.x[5].sin(), self.x[5].cos()]])
		F[:2, 5] = J.mv(Rot.mv(torch.Tensor([v*dt, 0])))

		A = torch.zeros(2, 3)
		A[0, 0] = 1
		A[0, 1] = self.x[3].sin() * self.x[4].tan()
		A[0, 2] = self.x[3].cos() * self.x[4].tan()
		A[1, 1] = self.x[3].cos()
		A[1, 2] = -self.x[3].sin()
		F[3:5, 6:9] = A*dt
		B = torch.Tensor([[1/2, 1/2], # v_l, v_r to v_forward
		                  [0, 0]])
		F[3, 3] = 1 + self.x[7]*self.x[3].sin()*self.x[4].tan()*dt
		F[3, 4] = 1 + (self.x[7]*self.x[3].sin() + self.x[8]*self.x[3].cos())*self.x[4].tan()*dt
		F[4, 3] = 1 - self.x[8]*self.x[3].sin()*dt
		G[:2, :2] = SO2.from_angle(self.x[5].unsqueeze(0)).as_matrix().mm(B)*dt

		# add Jacobian correction
		Q = torch.zeros_like(self.P)
		for i in range(G_cor.shape[0]):
			Q += (G+G_cor[i]).mm(self.Q).mm((G+G_cor[i]).t())
		self.P = F.mm(self.P).mm(F.t()) + Q

	def encoder2speed(self, u_odo, dt):
		res = self.calibration_parameters["Encoder resolution"]
		r_l = self.calibration_parameters["Encoder left wheel diameter"]
		r_r = self.calibration_parameters["Encoder right wheel diameter"]
		a = self.calibration_parameters["Encoder wheel base"]
		d_l = np.pi * r_l * u_odo[0] / res
		d_r = np.pi * r_r * u_odo[1] / res
		lin_speed = (d_l + d_r) / 2
		ang_speed = (d_l - d_r) / a
		return lin_speed/dt, ang_speed/dt






