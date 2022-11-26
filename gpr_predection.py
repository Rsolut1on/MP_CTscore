from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tools_gpr import *

# fit GPR
kernel = ConstantKernel(constant_value=.5, constant_value_bounds=(.1, 1e3)) * RBF(length_scale=1.5, \
                                                                                  length_scale_bounds=(1, 5))
noise = .75
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha=noise ** 2)

gpr_ols = GaussianProcessRegressor(optimizer=trust_region_optimizer, alpha=0.1, n_restarts_optimizer=3)
data_root = 'output/data_13_08422_real_12f_IDlist.mat'

data = sio.loadmat(data_root)
t_n = data['t_n']
t_n = t_n.astype(np.float64)
x_s_n = data['x_s_n']
t_p = data['t_p']
x_s_p = data['x_s_p']

z_s_origin = data['x_score']
z_t = data['t']
label = data['y']
z_s = preprocessing.scale(z_s_origin, axis=0, with_mean=True, with_std=True, copy=True)

z_t_n = z_t[label == 0]
z_t_n = z_t_n.astype(np.float64)
z_s_n = z_s[label == 0]
z_t_p = z_t[label == 1]
z_s_p = z_s[label == 1]

train_X, test_X, train_y, test_y = train_test_split(t_n, x_s_n, test_size=0.3, random_state=1)
train_x_z, test_x_z, train_y_z, test_y_z = train_test_split(z_t_n, z_s_n, test_size=.3, random_state=1)
# train_x_z, test_x_z, train_y_z, test_y_z = train_x_z[train_x_z<8], test_x_z[test_x_z<8], \
#                                            train_y_z[train_x_z<8], test_y_z[test_x_z<8]
train_x_z, test_x_z, train_y_z, test_y_z = train_x_z[7<train_x_z], test_x_z[7<test_x_z], \
                                           train_y_z[7<train_x_z], test_y_z[7<test_x_z]
train_x_z, test_x_z, train_y_z, test_y_z = train_x_z[train_x_z<29], test_x_z[test_x_z<29], \
                                           train_y_z[train_x_z<29], test_y_z[test_x_z<29]
# train_x_z, test_x_z, train_y_z, test_y_z = train_x_z[train_x_z>28], test_x_z[test_x_z>28], \
#                                            train_y_z[train_x_z>28], test_y_z[test_x_z>28]

# z_t_p, z_s_p = z_t_p[z_t_p<8], z_s_p[z_t_p<8]
z_t_p, z_s_p = z_t_p[7<z_t_p], z_s_p[7<z_t_p]
z_t_p, z_s_p = z_t_p[z_t_p<29], z_s_p[z_t_p<29]
# z_t_p, z_s_p = z_t_p[z_t_p>28], z_s_p[z_t_p>28]


# cat 5% noise
index = train_y_z.argsort(axis=0)
num_ct = len(index)

plt_test_x = np.arange(1, 100, .1).reshape(-1, 1)


train_x_z = train_x_z.reshape(-1, 1)
test_x_z = test_x_z.reshape(-1, 1)
train_y_z = train_y_z.ravel()
gpr.fit(train_x_z, train_y_z)
gpr_ols.fit(train_x_z, train_y_z)
mu, cov = gpr.predict(plt_test_x, return_cov=True)
mu_ols, cov_ols = gpr_ols.predict(plt_test_x, return_cov=True)
plt_test_y = mu.ravel()
plt_test_y_ols = mu_ols.ravel()
uncertainty = 3 * np.sqrt(np.diag(cov))  # 1.96 99.7%
uncertainty_in = 1.96 * np.sqrt(np.diag(cov))  # 1.96   95%c IC
print(max(uncertainty), max(uncertainty))
# plotting
plt.figure()
plt.xlim([0, 100])

## GPR
plt.scatter(train_x_z, train_y_z, label="training data", c="#CACACA", marker="o")  # 标准化数据
plt.scatter(test_x_z, test_y_z, label="test data", c="#A1C7E0", marker="o")
plt.scatter(z_t_p, z_s_p, label="severed data", c="#E3C75F", marker="*")


print('Train acc:{:.2f}'.format(gpr.score(train_X, train_y)))
print('Test acc:{:.2f}'.format(gpr.score(test_X, test_y)))
plt.xlabel('Time from the onset of initial symptoms(d)')
plt.ylabel('CT Score')
plt.legend(loc='best')
plt.show()

# norm_ctscore
plt.scatter(train_x_z, train_y_z, label="training data", c="#CACACA", marker="o")
plt.scatter(test_x_z, test_y_z, label="test data", c="#A1C7E0", marker="o")
plt.scatter(z_t_p, z_s_p, label="severed data", c="#E3C75F", marker="*")
plt.plot([0, 100], [0, 0], color="black", lw=.5, linestyle="--", alpha=0.5)
plt.xlabel('Time from the onset of initial symptoms(d)')
plt.ylabel('Z-score')
plt.legend(loc='best')
plt.show()

merics_roc_v2(train_x_z, train_y_z, test_x_z, test_y_z, z_t_p, z_s_p,
              [plt_test_x, plt_test_y, uncertainty_in, uncertainty], is_best=True)
merics_roc_v2(train_x_z, train_y_z, test_x_z, test_y_z, z_t_p, z_s_p,
              [plt_test_x, plt_test_y_ols, uncertainty_in, uncertainty])
