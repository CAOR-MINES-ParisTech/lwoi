
# Learning Wheel Odometry and IMU Errors for Localization
### Martin Brossard and Silvère Bonnabel

This repo containt the Python code for reproducing the results of the paper **Learning Wheel Odometry and IMU Errors for Localization**. Please follow the links to read the [paper](https://hal.archives-ouvertes.fr/hal-01874593/document).

## Installation & Pre-Requisites

 1. Install the master version of [PyTorch](https://pytorch.org/), the development version of [pyro](http://pyro.ai/), [liegroups](https://github.com/utiasSTARS/liegroups) and [progressbar](https://pypi.org/project/progressbar2/). Remaining packages are standard Python packages. All our code was running with Python 3.5.
 2.  Download data from one or two datasets (see below)
 3.  Clone the current repo
 ``` git clone https://github.com/Center-for-Robotics-MINES-ParisTech/gpkf ```


### University of Michigan North Campus Long-Term Vision and LiDAR Dataset

![nclt dataset image](https://github.com/CAOR-MINES-ParisTech/lwoi/blob/master/nclt.gif)

The _Segway_ dataset is described in the following [paper](http://journals.sagepub.com/doi/full/10.1177/0278364915614638):

- Nicholas Carlevaris-Bianco, Arash K. Ushani, and Ryan M. Eustice, _University of Michigan North Campus Long-Term Vision and Lidar Dataset_, International Journal of Robotics  Research, 2016.

Dataset can be downloaded following this [link](http://robots.engin.umich.edu/nclt/)  and extracted in `data/nclt`.

> - Training data: first 19 sequences
> - Cross-validation data: `2012-10-28`, `2012-11-04`, `2012-11-16`, `2012-11-17`
> - Testing data: `2012-12-01`, `2013-01-10`, `2013-02-23`, `2013-04-05` 

### Complex Urban LiDAR Data Set

![kaist dataset image](https://github.com/CAOR-MINES-ParisTech/lwoi/blob/master/urban16.gif)

The car dataset is based on the [paper](https://arxiv.org/abs/1803.06121)

- Jinyong Jeong, Younggun Cho, Young-Sik Shin, Hyunchul Roh, Ayoung Kim, _Complex Urban LiDAR Data Set_, 2018.

Dataset can be downloaded following this [link](http://irap.kaist.ac.kr/dataset/) and extracted in `data/kaist`.

- Training data: `urban00` to `urban11` and `campus00`
- Cross-validation data: `urban12`, `urban13`, `urban14`
- Testing data: `urban15`, `urban16`

## Training and Testing
 1. Modify setting and parameters if nessesary in `main_nclt.py` or `main_kaist.py`
 2. Run `main_nclt.py` or `main_kaist.py`

## Citing the paper

If you find this code useful for your research, please consider citing the following paper:

	@unpublished{brossard2018Learning,
	  Title          = {Learning Wheel Odometry and IMU Errors for Localization},
	  Author         = {Brossard, Martin and and Bonnabel Silvère},
	  Year           = {2019}
	}

##  License
For academic usage, the code is released under the permissive MIT license.

## Acknowledgements
We thank the authors of the University of Michigan North Campus Long-Term Vision and LiDAR Dataset and especially Arash \textsc{Ushani} for sharing their wheel encoder data log.

