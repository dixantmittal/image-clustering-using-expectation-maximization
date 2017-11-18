import numpy as np
from matplotlib import image
import matplotlib.pyplot as plt
from k_means import *
import scipy.stats as st
from tqdm import tqdm


def initialize_params(k, d):
    pi = np.random.rand(k)
    # normalize pi
    pi = pi / np.sum(pi)

    mew = np.random.randn(k, d) * 100

    # identity matrix
    sigma = np.identity(d) * 100
    sigma = np.asarray([sigma] * k)

    return pi, mew, sigma


def cal_expectation(data, pi, mew, sigma, k):
    n, d = data.shape
    gamma = np.zeros((k, n))

    for i in range(k):
        gamma[i] = pi[i] * st.multivariate_normal.pdf(data, mew[i], sigma[i])

    sum_gamma = np.sum(gamma, axis=0)
    gamma = gamma / sum_gamma
    return gamma


def cal_maximization(data, gamma, k):
    n, d = data.shape

    Nk = np.sum(gamma, axis=1)

    pi = Nk / n

    mew = np.zeros((k, d))

    for i in range(k):
        mew[i] = np.sum(gamma[i] * data.transpose()) / Nk[i]

    sigma = np.empty((k, d, d))
    for i in range(k):
        diff = data - mew[i]
        sigma[i] = np.dot(gamma[i] * diff.T, diff) / Nk[i]

    return pi, mew, sigma


def main():
    img_mat = image.imread('images/zebra.jpg')

    k = 2
    h, w, d = img_mat.shape
    img_mat = img_mat.reshape((h * w, d))

    pi, mew, sigma = initialize_params(k, d)

    mew = calculate_k_means(img_mat, k)

    pi_old, mew_old, sigma_old = pi, mew, sigma

    print('starting EM')
    for i in tqdm(range(100)):
        gamma = cal_expectation(img_mat, pi, mew, sigma, k)

        pi, mew, sigma = cal_maximization(img_mat, gamma, k)

        # until convergence
        if np.sum((mew - mew_old) ** 2) < 1e-10:
            break
        mew_old = mew

    prob = np.zeros((k, h * w))
    for i in range(k):
        prob[i] = pi[i] * st.multivariate_normal.pdf(img_mat, mew[i], sigma[i])

    assignment = np.argmax(prob, axis=0)
    # assignment = np.asarray(np.dstack((assignment, assignment, assignment)), dtype=np.float32) / (k - 1 + 1e-20)
    img_mat = img_mat.reshape((h, w, d))

    mask1 = np.asarray(img_mat * 255 * assignment.reshape((h, w, 1)), dtype=np.float32)
    plt.imshow(mask1)
    plt.show()

    assignment = -assignment + 1
    mask2 = np.asarray(img_mat * 255 * assignment.reshape((h, w, 1)), dtype=np.float32)
    plt.imshow(mask2, cmap=plt.cm.gray)
    plt.show()


if __name__ == '__main__':
    main()
