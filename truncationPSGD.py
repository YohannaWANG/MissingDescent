import numpy as np
import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from tqdm import tqdm

import config
p = config.setup()

class MeanEst():
    def __init__(self):
        self.affine_transform = True

    def PairwiseEmp(self, X, pair_idx, M, batch_size):
        mu_hatS_lst = []
        Sig_hatS_lst = []
        O_aff_lst = []
        rand_sample_idx = []
        n_lst = []
        for i in range(X.shape[1]):
            O = X[pair_idx[str(i)], :][:, [i]]

            mu_hatS = torch.mean(O, axis=0)
            X_pair = O - mu_hatS
            n, d = X_pair.shape[0], 1
            A = X_pair.reshape(n, d, 1)
            B = torch.transpose(A, dim0=2, dim1=1)
            Sig_hatS = torch.mean(torch.matmul(A, B), axis=0)

            if self.affine_transform == True:
                O_aff = (1 / torch.sqrt(Sig_hatS)).matmul(X_pair.t()).t()
            else:
                O_aff = O

            mu_hatS_lst.append(mu_hatS.reshape(-1, 1))
            Sig_hatS_lst.append(Sig_hatS)
            O_aff_lst.append(O_aff)
            rand_sample_idx.append(np.random.choice(range(0, n), (M, batch_size)))
            n_lst.append(n)

        n_max = np.max(n_lst)

        for i in range(len(O_aff_lst)):
            n = n_lst[i]
            nan_tensor = torch.zeros((n_max - n, 1))
            nan_tensor[:, :] = float('nan')
            O_aff_lst[i] = torch.cat((O_aff_lst[i], nan_tensor), axis=0)

        return torch.stack(mu_hatS_lst), torch.stack(Sig_hatS_lst), torch.stack(O_aff_lst), np.stack(
            rand_sample_idx).transpose((1, 0, 2))  # np.stack(rand_sample_idx).T

    def Pair_IDX(self, X, cond_lower, cond_upper):
        pair_idx = {}
        cond_lower_lst = []
        cond_upper_lst = []
        for i in range(X.shape[1]):
            pair_idx[str(i)] = torch.where(~torch.isnan(X[:, i]) == True)[0]
            cond_lower_lst.append(torch.tensor([cond_lower[i]]))
            cond_upper_lst.append(torch.tensor([cond_upper[i]]))

        return pair_idx, torch.stack(cond_lower_lst), torch.stack(cond_upper_lst)

    def SGD(self, M, batch_size, lamb, X, cond_lower, cond_upper, show_progress_bar=True, display_per_iter=100):

        pair_idx, pair_cond_lower, pair_cond_upper = self.Pair_IDX(X, cond_lower, cond_upper)
        mu_hatS_lst, Sig_hatS_lst, O_aff_lst, rand_sample_idx = self.PairwiseEmp(X, pair_idx, M, batch_size)

        if self.affine_transform == True:
            w = [torch.stack([torch.cat((torch.eye(1).reshape(-1), torch.zeros(1))) for i in range(len(pair_idx))])]
        else:
            T_hatS_lst = 1 / Sig_hatS_lst
            v_hatS_lst = T_hatS_lst.matmul(mu_hatS_lst)
            w = [torch.cat(
                (T_hatS_lst.reshape((Sig_hatS_lst.shape[0], 4)), v_hatS_lst.reshape((mu_hatS_lst.shape[0], 2))), dim=1)]

        grad_lst = []

        if show_progress_bar == True:
            loop_list = tqdm(range(1, M + 1))
        else:
            loop_list = range(1, M + 1)

        for i in loop_list:
            idx = rand_sample_idx[i - 1].T
            x = O_aff_lst[range(0, O_aff_lst.shape[0]), idx, :].transpose(dim0=1, dim1=0).transpose(dim0=2, dim1=1)

            nu = 1 / lamb
            grad = self.GradientEstimation(x, w[-1], mu_hatS_lst, Sig_hatS_lst, pair_cond_lower, pair_cond_upper,
                                           batch_size)
            grad_lst.append(torch.sum(grad[0, :] ** 2))
            r = w[-1] - nu * grad
            # w.append(self.ProjectToDomain(r, r_star))
            w.append(r)

            if show_progress_bar == False and (i + 1) % display_per_iter == 0:
                print('Iteration:', i + 1)

        w = torch.stack(w)
        N = min(int(w.shape[0] / 2), 1000)
        w_bar = torch.mean(w[-N:], axis=0)
        T_bar_lst = w_bar[:, :1].reshape((w_bar.shape[0], 1, 1))
        v_bar_lst = w_bar[:, 1:].reshape((w_bar.shape[0], 1, 1))

        Sig_bar = 1 / T_bar_lst
        mu_bar = Sig_bar.matmul(v_bar_lst)

        if self.affine_transform == True:
            Sig_hatS_half = torch.sqrt(Sig_hatS_lst)

            Sig_hat = Sig_hatS_half.matmul(Sig_bar).matmul(torch.transpose(Sig_hatS_half, dim0=2, dim1=1))
            mu_hat = Sig_hatS_half.matmul(mu_bar) + mu_hatS_lst
        else:
            Sig_hat = Sig_bar
            mu_hat = mu_bar

        return (Sig_hat, mu_hat, torch.stack(grad_lst), w)

    def MS(self, y, mu_hatS, Sig_hatS, pair_cond_lower, pair_cond_upper):
        if self.affine_transform == True:
            Sig_hatS_half = torch.sqrt(Sig_hatS)
            y_prime = Sig_hatS_half.matmul(y) + mu_hatS  # .reshape((y.shape[0], y.shape[1]))
        else:
            y_prime = y

        pair_cond_lower = pair_cond_lower.reshape((pair_cond_lower.shape[0], pair_cond_lower.shape[1], 1))
        pair_cond_upper = pair_cond_upper.reshape((pair_cond_upper.shape[0], pair_cond_upper.shape[1], 1))
        cond = torch.logical_and(y_prime > pair_cond_lower, y_prime < pair_cond_upper)
        # cond = ~torch.logical_or(cond[:,0,:], cond[:,1,:])
        cond = ~cond[:, 0, :]
        idx_accept = torch.where(cond == True)
        return idx_accept

    def GradientEstimation(self, x, w, mu_hatS, Sig_hatS, pair_cond_lower, pair_cond_upper, batch_size):
        T_lst, v_lst = w[:, :1].reshape((w.shape[0], 1, 1)), w[:, 1:].reshape(w.shape[0], 1, 1)
        Sigma = 1 / T_lst
        Sigma_half = torch.sqrt(Sigma)
        mu = Sigma.matmul(v_lst)

        y = torch.zeros((w.shape[0], 1, batch_size))
        marker = torch.zeros(w.shape[0], batch_size)

        while True:
            y_sample = mu + Sigma_half.matmul(torch.randn((w.shape[0], 1, batch_size)))
            idx_accept_row, idx_accept_row_col = self.MS(y_sample, mu_hatS, Sig_hatS, pair_cond_lower, pair_cond_upper)
            if len(idx_accept_row) > 0:
                y[idx_accept_row, :, idx_accept_row_col] = y_sample[idx_accept_row, :, idx_accept_row_col]
                marker[idx_accept_row, idx_accept_row_col] = 1
            if torch.sum(marker) == marker.shape[0] * marker.shape[1]:
                break

        x = x.transpose(dim0=2, dim1=1)
        x = x.reshape((x.shape[0], x.shape[1], 1, 1))
        y = y.transpose(dim0=2, dim1=1)
        y = y.reshape((y.shape[0], y.shape[1], 1, 1))

        M = 0.5 * torch.mean(x.matmul(torch.transpose(x, dim0=3, dim1=2)), dim=1) - 0.5 * torch.mean(
            y.matmul(torch.transpose(y, dim0=3, dim1=2)), dim=1)
        m = -torch.mean(x, dim=1) + torch.mean(y, dim=1)
        return torch.cat((M.reshape(w.shape[0], 1), m.reshape((w.shape[0], 1))), axis=1)

    def LearnSigma(self, Sigma_hat_lst, mu_hat_lst, dim):
        pair_idx = {}
        k = 0
        for i in range(dim):
            for j in range(i + 1, dim):
                pair_idx[str(i) + str(j)] = k
                k += 1

        Sig_learned = cp.Variable((dim, dim), symmetric=True)
        Sig_hat_ij = cp.Parameter((int(dim * (dim - 1) / 2), 4))

        constraints = [Sig_learned >> 0]  # Positive semidefinite constraint
        constraints += [
            Sig_learned[:, [i, j]][[i, j], :] << (1 + 0.05) * cp.reshape(Sig_hat_ij[pair_idx[str(i) + str(j)]], (2, 2))
            for i in range(dim) for j in range(i + 1, dim)
        ]
        constraints += [
            Sig_learned[:, [i, j]][[i, j], :] >> (1 - 0.05) * cp.reshape(Sig_hat_ij[pair_idx[str(i) + str(j)]], (2, 2))
            for i in range(dim) for j in range(i + 1, dim)
        ]

        prob = cp.Problem(cp.Minimize(0), constraints)
        # prob.solve()
        cvxpylayer = CvxpyLayer(prob, parameters=[Sig_hat_ij], variables=[Sig_learned])

        # solve the problem
        solution, = cvxpylayer(Sigma_hat_lst.reshape(-1, 4))

        return solution

    def Train(self, M, batch_size, lamb, X, cond_lower, cond_upper,
              affine_transform=True, show_progress_bar=True, display_per_iter=100):
        self.affine_transform = affine_transform
        Sigma_hat_lst, mu_hat_lst, grad, w = self.SGD(M, batch_size, lamb, X, cond_lower, cond_upper,
                                                      show_progress_bar=show_progress_bar,
                                                      display_per_iter=display_per_iter)
        return Sigma_hat_lst, mu_hat_lst, grad, w


class CovarianceEst():
    def __init__(self):
        self.affine_transform = True
        self.rand_batch_idx = np.vectorize(self.rand_one_batch_idx, signature='(),(),()->(m,k)')

    def rand_one_batch_idx(self, n, M, batch_size):
        return np.random.choice(range(0, n), (M, batch_size))

    def PairwiseEmp(self, X, pair_idx, M, batch_size, init_mu=None, init_var=None):
        mu_hatS_lst = []
        Sig_hatS_lst = []
        O_aff_lst = []
        mu_init_lst = []
        var_init_lst = []
        n_lst = []
        for i in range(X.shape[1]):
            for j in range(i + 1, X.shape[1]):
                O = X[pair_idx[str(i) + str(j)], :][:, [i, j]]

                mu_hatS = torch.mean(O, axis=0)
                X_pair = O - mu_hatS
                n, d = X_pair.shape[0], 2
                A = X_pair.reshape(n, d, 1)
                B = torch.transpose(A, dim0=2, dim1=1)
                Sig_hatS = torch.mean(torch.matmul(A, B), axis=0)

                if self.affine_transform == True:
                    O_aff = torch.inverse(torch.linalg.cholesky(Sig_hatS)).matmul(X_pair.t()).t()
                else:
                    O_aff = O

                mu_hatS_lst.append(mu_hatS.reshape(-1, 1))
                Sig_hatS_lst.append(Sig_hatS)
                O_aff_lst.append(O_aff)

                if init_mu != None:
                    mu_init_lst.append(torch.tensor([init_mu[i], init_mu[j]]).reshape(-1, 1))
                if init_var != None:
                    var_init_lst.append(torch.tensor([init_var[i], init_var[j]]).reshape(-1, 1))
                n_lst.append(n)

        n_max = np.max(n_lst)

        for i in range(len(O_aff_lst)):
            n = n_lst[i]
            nan_tensor = torch.zeros((n_max - n, 2))
            nan_tensor[:, :] = float('nan')
            O_aff_lst[i] = torch.cat((O_aff_lst[i], nan_tensor), axis=0)

        r = [torch.stack(mu_hatS_lst), torch.stack(Sig_hatS_lst), torch.stack(O_aff_lst), n_lst]
        if init_mu != None:
            r = r + [torch.stack(mu_init_lst)]
        else:
            r = r + [None]
        if init_var != None:
            r = r + [torch.stack(var_init_lst)]
        else:
            r = r + [None]
        return r

    def Pair_IDX(self, X, cond_lower, cond_upper):
        pair_idx = {}
        cond_lower_lst = []
        cond_upper_lst = []
        for i in range(X.shape[1]):
            for j in range(i + 1, X.shape[1]):
                pair_idx[str(i) + str(j)] = \
                torch.where(torch.logical_and(~torch.isnan(X[:, i]), ~torch.isnan(X[:, j])) == True)[0]
                cond_lower_lst.append(torch.tensor([cond_lower[i], cond_lower[j]]))
                cond_upper_lst.append(torch.tensor([cond_upper[i], cond_upper[j]]))

        return pair_idx, torch.stack(cond_lower_lst), torch.stack(cond_upper_lst)

    def SGD(self, M, batch_size, lamb, X, cond_lower, cond_upper,
            init_mu=None, init_var=None, show_progress_bar=True, display_per_iter=100):

        pair_idx, pair_cond_lower, pair_cond_upper = self.Pair_IDX(X, cond_lower, cond_upper)
        mu_hatS_lst, Sig_hatS_lst, O_aff_lst, n_lst, mu_init_lst, var_init_lst = self.PairwiseEmp(X, pair_idx, M,
                                                                                                  batch_size, init_mu,
                                                                                                  init_var)

        if self.affine_transform == True:
            if init_var == None:
                T_hatS_lst = torch.stack([torch.eye(2) for i in range(len(pair_idx))])
            else:
                Sig_hatS_half = torch.linalg.cholesky(Sig_hatS_lst)
                Sig_init = Sig_hatS_lst * (1 - torch.eye(2)) + torch.eye(2) * var_init_lst
                Sig_init_aff = torch.inverse(Sig_hatS_half).matmul(Sig_init).matmul(
                    torch.inverse(torch.transpose(Sig_hatS_half, dim0=2, dim1=1)))
                T_hatS_lst = torch.inverse(Sig_init_aff)

            if init_mu == None:
                v_hatS_lst = torch.stack([torch.zeros(2) for i in range(len(pair_idx))])
            else:
                Sig_hatS_half = torch.linalg.cholesky(Sig_hatS_lst)
                m_init = torch.inverse(Sig_hatS_half).matmul(mu_init_lst - mu_hatS_lst)
                v_hatS_lst = T_hatS_lst.matmul(m_init)
            w = [torch.cat(
                (T_hatS_lst.reshape((Sig_hatS_lst.shape[0], 4)), v_hatS_lst.reshape((mu_hatS_lst.shape[0], 2))), dim=1)]
        else:
            if init_var == None:
                T_hatS_lst = torch.inverse(Sig_hatS_lst)
            else:
                T_hatS_lst = torch.inverse(Sig_hatS_lst * (1 - torch.eye(2)) + torch.eye(2) * var_init_lst)

            if init_mu == None:
                v_hatS_lst = T_hatS_lst.matmul(mu_hatS_lst)
            else:
                v_hatS_lst = T_hatS_lst.matmul(mu_init_lst)
            w = [torch.cat(
                (T_hatS_lst.reshape((Sig_hatS_lst.shape[0], 4)), v_hatS_lst.reshape((mu_hatS_lst.shape[0], 2))), dim=1)]

        M_mini = 5000
        rand_sample_idx = self.rand_batch_idx(n_lst, M=M_mini, batch_size=batch_size).transpose((1, 0, 2))
        grad_lst = []

        if show_progress_bar == True:
            loop_list = tqdm(range(1, M + 1))
        else:
            loop_list = range(1, M + 1)

        for i in loop_list:
            idx = rand_sample_idx[(i - 1) % M_mini].T
            if i == M_mini:
                rand_sample_idx = None  # clear memory
                rand_sample_idx = self.rand_batch_idx(n_lst, M=M_mini, batch_size=batch_size).transpose((1, 0, 2))
            # idx = self.rand_batch_idx(n_lst, batch_size).T

            x = O_aff_lst[range(0, O_aff_lst.shape[0]), idx, :].transpose(dim0=1, dim1=0).transpose(dim0=2, dim1=1)

            nu = 1 / lamb
            grad = self.GradientEstimation(x, w[-1], mu_hatS_lst, Sig_hatS_lst, pair_cond_lower, pair_cond_upper,
                                           batch_size)

            grad_lst.append(torch.sum(grad[0, :] ** 2))
            r = w[-1] - nu * grad
            # w.append(self.ProjectToDomain(r, r_star))
            w.append(r)

            if show_progress_bar == False and (i + 1) % display_per_iter == 0:
                print('Iteration:', i + 1)

        w = torch.stack(w)
        N = min(int(w.shape[0] / 2), 1000)
        w_bar = torch.mean(w[-N:], axis=0)
        T_bar_lst = w_bar[:, :4].reshape((w_bar.shape[0], 2, 2))
        v_bar_lst = w_bar[:, 4:].reshape((w_bar.shape[0], 2, 1))

        Sig_bar = torch.inverse(T_bar_lst)
        mu_bar = Sig_bar.matmul(v_bar_lst)

        if self.affine_transform == True:
            Sig_hatS_half = torch.linalg.cholesky(Sig_hatS_lst)

            Sig_hat = Sig_hatS_half.matmul(Sig_bar).matmul(torch.transpose(Sig_hatS_half, dim0=2, dim1=1))
            mu_hat = Sig_hatS_half.matmul(mu_bar) + mu_hatS_lst
        else:
            Sig_hat = Sig_bar
            mu_hat = mu_bar

        return (Sig_hat, mu_hat, torch.stack(grad_lst), w)

    def MS(self, y, mu_hatS, Sig_hatS, pair_cond_lower, pair_cond_upper):
        if self.affine_transform == True:
            Sig_hatS_half = torch.linalg.cholesky(Sig_hatS)
            y_prime = Sig_hatS_half.matmul(y) + mu_hatS  # .reshape((y.shape[0], y.shape[1]))
        else:
            y_prime = y

        pair_cond_lower = pair_cond_lower.reshape((pair_cond_lower.shape[0], pair_cond_lower.shape[1], 1))
        pair_cond_upper = pair_cond_upper.reshape((pair_cond_upper.shape[0], pair_cond_upper.shape[1], 1))
        cond = torch.logical_and(y_prime > pair_cond_lower, y_prime < pair_cond_upper)
        cond = ~torch.logical_or(cond[:, 0, :], cond[:, 1, :])
        idx_accept = torch.where(cond == True)
        return idx_accept

    def GradientEstimation(self, x, w, mu_hatS, Sig_hatS, pair_cond_lower, pair_cond_upper, batch_size):
        T_lst, v_lst = w[:, :4].reshape((w.shape[0], 2, 2)), w[:, 4:].reshape(w.shape[0], 2, 1)
        Sigma = torch.inverse(T_lst)
        Sigma_half = torch.linalg.cholesky(Sigma)
        mu = Sigma.matmul(v_lst)

        y = torch.zeros((w.shape[0], 2, batch_size))
        marker = torch.zeros(w.shape[0], batch_size)

        while True:
            y_sample = mu + Sigma_half.matmul(torch.randn((w.shape[0], 2, batch_size)))

            idx_accept_row, idx_accept_row_col = self.MS(y_sample, mu_hatS, Sig_hatS, pair_cond_lower, pair_cond_upper)
            if len(idx_accept_row) > 0:
                y[idx_accept_row, :, idx_accept_row_col] = y_sample[idx_accept_row, :, idx_accept_row_col]
                marker[idx_accept_row, idx_accept_row_col] = 1
            if torch.sum(marker) == marker.shape[0] * marker.shape[1]:
                break

        x = x.transpose(dim0=2, dim1=1)
        x = x.reshape((x.shape[0], x.shape[1], 2, 1))
        y = y.transpose(dim0=2, dim1=1)
        y = y.reshape((y.shape[0], y.shape[1], 2, 1))

        M = 0.5 * torch.mean(x.matmul(torch.transpose(x, dim0=3, dim1=2)), dim=1) - 0.5 * torch.mean(
            y.matmul(torch.transpose(y, dim0=3, dim1=2)), dim=1)
        m = -torch.mean(x, dim=1) + torch.mean(y, dim=1)
        return torch.cat((M.reshape(w.shape[0], 4), m.reshape((w.shape[0], 2))), axis=1)

    def LearnSigma(self, Sigma_hat_lst, mu_hat_lst, dim):
        pair_idx = {}
        k = 0
        for i in range(dim):
            for j in range(i + 1, dim):
                pair_idx[str(i) + str(j)] = k
                k += 1

        Sig_learned = cp.Variable((dim, dim), symmetric=True)
        Sig_hat_ij = cp.Parameter((int(dim * (dim - 1) / 2), 4))
        # The operator >> denotes matrix inequality.
        constraints = [Sig_learned >> 0]  # Positive semidefinite constraint
        constraints += [
            Sig_learned[:, [i, j]][[i, j], :] << (1 + 0.05) * cp.reshape(Sig_hat_ij[pair_idx[str(i) + str(j)]], (2, 2))
            for i in range(dim) for j in range(i + 1, dim)
        ]
        constraints += [
            Sig_learned[:, [i, j]][[i, j], :] >> (1 - 0.05) * cp.reshape(Sig_hat_ij[pair_idx[str(i) + str(j)]], (2, 2))
            for i in range(dim) for j in range(i + 1, dim)
        ]

        prob = cp.Problem(cp.Minimize(0), constraints)
        # prob.solve()
        cvxpylayer = CvxpyLayer(prob, parameters=[Sig_hat_ij], variables=[Sig_learned])
        # solve the problem
        solution, = cvxpylayer(Sigma_hat_lst.reshape(-1, 4))
        return solution

    def Train(self, M, batch_size, lamb, X, cond_lower, cond_upper, init_mu=None, init_var=None,
              affine_transform=True, show_progress_bar=True, display_per_iter=100):
        self.affine_transform = affine_transform
        Sigma_hat_lst, mu_hat_lst, grad, w = self.SGD(M, batch_size, lamb, X, cond_lower, cond_upper,
                                                      init_mu=init_mu, init_var=init_var,
                                                      show_progress_bar=show_progress_bar,
                                                      display_per_iter=display_per_iter)
        # Sigma_learned = self.LearnSigma(X, Sigma_hat_lst, mu_hat_lst)
        return Sigma_hat_lst, mu_hat_lst, grad, w

    def learn_v_prime(self, v, r2):
        b = cp.Variable((1, 2))
        v_parms = cp.Parameter(v.shape)
        constraints = [cp.constraints.nonpos.NonPos(cp.norm(b, 2) - r2)]
        obj = cp.sum_squares(b - v_parms)

        prob = cp.Problem(cp.Minimize(obj), constraints)
        cvxpylayer = CvxpyLayer(prob, parameters=[v_parms], variables=[b])

        solution, = cvxpylayer(v)

        return solution

    def learn_T_prime(self, T, r3, lamb):
        T_prime = cp.Variable((2, 2))
        T_parms = cp.Parameter(T.shape)
        I_parms = cp.Parameter(T.shape)
        constraints = [T_prime >> r3 * I_parms]
        obj = cp.sum_squares(T_prime - T_parms) + lamb * cp.sum_squares(I_parms - T_prime)
        prob = cp.Problem(cp.Minimize(obj), constraints)
        cvxpylayer = CvxpyLayer(prob, parameters=[T_parms, I_parms], variables=[T_prime])
        I = torch.eye(2)
        solution, = cvxpylayer(T, I)

        opt_val = torch.sum((solution - T) ** 2) + lamb * torch.sum((I - solution) ** 2)

        return solution, opt_val

    def ProjectToDomain(self, r, r_star):
        T, v = r[:, :4].reshape(2, 2), r[:, 4:]
        r1, r2, r3 = r_star, r_star, 1 / r_star

        v_prime = self.learn_v_prime(v, r1)

        lamb = 1
        cond_lst = []
        T_prime_lst = []
        opt_val_lst = []
        for i in range(10):
            T_prime, opt_value = self.learn_T_prime(T, r3, lamb)

            cond = torch.sum((torch.eye(2) - T_prime) ** 2)
            cond_lst.append(cond)
            T_prime_lst.append(T_prime)
            opt_val_lst.append(opt_value)

            lamb = lamb / 2

        cond_lst = torch.tensor(cond_lst)
        T_prime_lst = torch.stack(T_prime_lst)
        opt_val_lst = torch.tensor(opt_val_lst)

        T_prime_lst = T_prime_lst[cond_lst <= r2 ** 2]
        opt_val_lst = opt_val_lst[cond_lst <= r2 ** 2]

        return torch.cat((T_prime_lst[torch.argmin(opt_val_lst)].reshape(-1), v_prime.reshape(-1))).reshape(1, -1)


def LearnSigma(Sigma_hat_lst):

    dim = p.d
    pair_idx = {}
    k = 0
    for i in range(dim):
        for j in range(i + 1, dim):
            pair_idx[str(i) + str(j)] = k
            k += 1

    # Define and solve the CVXPY problem.
    # Create a symmetric matrix variable.
    Sigma_learned = cp.Variable((dim, dim), symmetric=True)
    # The operator >> denotes matrix inequality.
    constraints = [Sigma_learned >> 0]  # Positive semidefinite constraint
    L = cp.sum([cp.sum((Sigma_learned[:, [i, j]][[i, j], :] - Sigma_hat_lst[pair_idx[str(i) + str(j)]]) ** 2) for i in
                range(dim) for j in range(i + 1, dim)])
    prob = cp.Problem(cp.Minimize(L), constraints)
    prob.solve()

    return torch.tensor(Sigma_learned.value)