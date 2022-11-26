import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from scipy.optimize import least_squares, leastsq


def y(x, noise_sigma=0.0):
    x = np.asarray(x)
    y = np.cos(x) + np.random.normal(0, noise_sigma, size=x.shape)
    return y.tolist()


def line_total_least_squares(x_in, y_in):
    n = len(x_in)

    x_m = np.sum(x_in) / n
    y_m = np.sum(y_in) / n

    # Calculate the x~ and y~
    x1 = x_in - x_m
    y1 = y_in - y_m

    # Create the matrix array
    X = np.vstack((x1, y1))
    X_t = np.transpose(X)

    # Finding A_T_A and it's Find smallest eigenvalue::
    prd = np.dot(X, X_t)
    W, V = np.linalg.eig(prd)
    small_eig_index = W.argmin()
    a, b = V[:, small_eig_index]

    # Compute C:
    c = (-1 * a * x_m) + (-1 * b * y_m)

    return a, b, c


def merics_roc(test_x_z, test_y_z, z_t_p, z_s_p):
    # [4.7, -1.5]
    cur_k = 4.7
    fps, tps = [], []
    for i in range(62):
        cur_k -= .1
        fp = np.sum(test_y_z > cur_k)
        fn = np.sum(z_s_p < cur_k)
        tp = np.sum(z_s_p > cur_k)
        tn = np.sum(test_y_z < cur_k)
        fps.append(fp)
        tps.append(tp)
    num = len(test_y_z) + len(z_s_p)
    fpr = np.array(fps) / len(z_s_p)
    tpr = np.array(tps) / len(z_s_p)

    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="#196774",
        lw=lw,
        label="L-BFGS: AUC=%0.2f" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("$N_{test}$=39, $N_{severed}$=39")
    plt.legend(loc="lower right")
    # plt.show()


def merics_roc_v2(train_x_z, train_y_z, test_x_z, test_y_z, z_t_p, z_s_p, plt_test, is_best=False, uncertain_in=None):
    plt_x, plt_y = plt_test[0], plt_test[1]
    uncertain_in, uncertain = plt_test[2], plt_test[2]
    plt_y = plt_y.reshape([-1, 1])
    for i in range(len(test_x_z)):
        index = test_x_z[i]
        index_plt = int(index - 1) * 10
        test_y_z[i] -= plt_y[index_plt]
        index = z_t_p[i]
        index_plt = int(index - 1) * 10
        z_s_p[i] -= plt_y[index_plt]
    for i in range(len(train_x_z)):
        index = train_x_z[i]
        if index > 98:
            continue
        index_plt = int(index - 1) * 10
        train_y_z[i] -= plt_y[index_plt]
    if is_best:
        # test_y_z = abs(test_y_z)
        # z_s_p = abs(z_s_p)
        plt.scatter(train_x_z, train_y_z, label="training data", c="#CACACA", marker="o")
        plt.scatter(test_x_z, test_y_z, label="test data", c="#A1C7E0", marker="o")
        plt.scatter(z_t_p, z_s_p, label="severed data", c="#E3C75F", marker="*")
        plt.plot([0, 100], [0, 0], color="black", lw=.5, linestyle="--", alpha=0.5)
        plt.xlabel('Time from the onset of initial symptoms(d)')
        plt.ylabel('Z-score')
        plt.legend(loc='best')
        # plt.savefig('../figs/z-score.pdf')
        plt.show()
    # ROC
    # [2.1 -0.6]; 标准化后：[-0.5, 4.7]; I:[-0.289, 2.6]
    cur_k = max(max(test_y_z), max(z_s_p))
    fps, tps = [], []
    # V1 maybe wrong
    #     for i in range(520):
    #         cur_k -= .01
    #         # cur_k = 0
    #         fp = np.sum(test_y_z > cur_k)
    #         fn = np.sum(z_s_p < cur_k)
    #         tp = np.sum(z_s_p > cur_k)
    #         tn = np.sum(test_y_z < cur_k)
    #         fps.append(fp)
    #         tps.append(tp)
    #     num = len(test_y_z) + len(z_s_p)
    #     fpr = np.array(fps) / len(z_s_p)
    #     tpr = np.array(tps) / len(z_s_p)
    # V2
    while (min(min(test_y_z), min(z_s_p)) < cur_k):
        # for i in range(600):
        cur_k -= .01
        # cur_k = 0
        # test_y_z = abs(test_y_z)
        # z_s_p = abs(z_s_p)

        fp = np.sum(test_y_z > cur_k)
        fn = np.sum(z_s_p < cur_k)
        tp = np.sum(z_s_p > cur_k)
        tn = np.sum(test_y_z < cur_k)
        fps.append(fp)
        tps.append(tp)
    num = len(test_y_z) + len(z_s_p)
    fpr = np.array(fps) / len(test_y_z)
    tpr = np.array(tps) / len(z_s_p)

    roc_auc = auc(fpr, tpr)

    # plt.figure()
    lw = 2
    if is_best:
        plt.plot(
            fpr,
            tpr,
            color="#196774",
            lw=lw,
            label="L-BFGS: AUC=%0.2f" % roc_auc,
        )
        np.savez('roc_best_s2.npz', fpr=fpr,tpr=tpr)
    else:
        plt.plot(
            fpr,
            tpr,
            color="#196774",
            linestyle=':',
            lw=lw,
            label="    OLS: AUC=%0.2f" % roc_auc,
        )
        np.savez('roc_OLS_s2.npz', fpr=fpr, tpr=tpr)
        plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("$N_{test}$=%s, $N_{severed}$=%s" % (len(test_y_z), len(z_s_p)))
        plt.legend(loc="lower right")
        # plt.savefig('../figs/ROC_stage3.pdf')
        plt.show()


def plot_id_list(ids_list, date, label, score):
    for i in range(99):
        patient_id = i + 1
        date_p = date[label == 1]
        score_p = score[label == 1]
        p_x = date_p[ids_list[label == 1] == patient_id]
        p_y = score_p[ids_list[label == 1] == patient_id]
        plt.plot(p_x, p_y, c='#E3C75F', alpha=0.3)

        date_n = date[label == 0]
        score_n = score[label == 0]
        p_x = date_n[ids_list[label == 0] == patient_id]
        p_y = score_n[ids_list[label == 0] == patient_id]
        plt.plot(p_x, p_y, c='#CACACA', alpha=0.3)


def trust_region_optimizer(obj_func, initial_theta, bounds):
    trust_region_method = leastsq(1 / obj_func, initial_theta, bounds, method='trf')
    return (trust_region_method.x, trust_region_method.fun)
