import numpy as np
from matplotlib import image
import matplotlib.pyplot as plt
from tqdm import tqdm


def calculate_k_means(data, k):
    n, d = data.shape

    # random initialize means
    mew_old = mew = np.random.randn(k, d)

    # repeatedly perform assignment and mean update
    # hard limit is 50
    print('starting k-means')
    for x in tqdm(range(50)):

        # assignment
        assignment = np.argmin(np.asarray([np.sum((data - mew[a]) ** 2, axis=1) for a in range(k)]), axis=0)

        # mean update
        mew = np.asarray([np.sum((assignment == a) * (data.transpose()), axis=1) / (
            np.sum(assignment == a) + 1e-8) for a in range(k)])

        # until convergence
        if np.sum((mew - mew_old) ** 2) < 1e-10:
            break
        mew_old = mew

    print("converged at iteration: ", x)

    return mew


def k_means_clustering(data, k):
    # find out k means
    mew = calculate_k_means(data, k)
    print("converged mew: \n", mew)

    # assign clusters
    assignment = np.argmin(np.asarray([np.sum((data - mew[a]) ** 2, axis=1) for a in range(k)]), axis=0)

    return assignment


def main():
    img_mat = image.imread('images/cow.jpg')
    h, w, d = img_mat.shape
    k = 2
    img_mat = img_mat.reshape((h * w, d))
    assignment = k_means_clustering(img_mat, k).reshape(h, w)

    # duplicate to create image to RGB
    assignment = np.asarray(np.dstack((assignment, assignment, assignment)), dtype=np.float32) / (k - 1 + 1e-20)
    plt.imshow(assignment)
    plt.show()


if __name__ == '__main__':
    main()
