import argparse
from main_kaist import launch

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='GP nclt')
	parser.add_argument('--nclt', type=bool, default=True)
	parser.add_argument('--path_data_base', type=str, default="/media/mines/DATA/NCLT/")
	parser.add_argument('--path_data_save', type=str, default="data/nclt/")
	parser.add_argument('--path_results', type=str, default="results/nclt/")
	parser.add_argument('--path_temp', type=str, default="temp/nclt/")

	# data extraction
	parser.add_argument('--y_diff_odo_fog_threshold', type=float, default=0.15)
	parser.add_argument('--y_diff_imu_threshold', type=float, default=0.4)

	# model parameters
	parser.add_argument('--delta_t', type=float, default=0.01)
	parser.add_argument('--Delta_t', type=float, default=1)
	parser.add_argument('--num_inducing_point', type=int, default=100)
	parser.add_argument('--kernel_dim', type=int, default=20)

	# optimizer parameters
	parser.add_argument('--epochs', type=int, default=100)
	parser.add_argument('--lr', type=float, default=0.01)
	parser.add_argument('--lr_decay', type=float, default=0.999)
	parser.add_argument('--compute_normalize_factors', type=bool, default=True)
	parser.add_argument('--compare', type=str, default="model")

	args = parser.parse_args()
	launch(args)

