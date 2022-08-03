
import torch
import numpy as np

from time import time
from data import synthetic_data
from utils_missing import produce_NA
from truncationPSGD import MeanEst, CovarianceEst, LearnSigma

import config
p = config.setup()

print('PyTorch version', torch.__version__)
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print('Use ***GPU***')
    print(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024, 'GB')
else:
    print('Use CPU')
    torch.set_default_tensor_type('torch.FloatTensor')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    """ Generate synthetic data """
    X_syn, N, d, mu, Sigma = synthetic_data()

    """ Self-masking missingness"""
    X_miss_selfmasked = produce_NA(X_syn, p_miss=0.4, mecha="MNAR", opt="selfmasked")
    percentile = -0.4
    cond_lower = mu + percentile * p.sqrt(np.diag(Sigma))
    cond_upper = [10000.0] * d

    X_mnar_selfmasked = X_miss_selfmasked['X_incomp']
    R_mnar_selfmasked = X_miss_selfmasked['mask']
    print("Percentage of generated missing values: ", (R_mnar_selfmasked.sum()).numpy()/np.prod(R_mnar_selfmasked.size())*100, " %")

    result_Sigma_pairwise = []
    result_mu = []
    time_lst = []

    n = p.n
    start = time()
    print('Estimating mean...')
    model_mu = MeanEst()
    var_hat_lst, mu_hat_lst, _, _ = model_mu.Train(M=200000,
                                                   batch_size=20,
                                                   lamb=1000,
                                                   X=torch.tensor(X_mnar_selfmasked, device=device).float()[:n, :],
                                                   cond_lower=torch.tensor(list(cond_lower)),
                                                   cond_upper=torch.tensor(list(cond_upper)),
                                                   affine_transform=True,
                                                   display_per_iter=200)

    print('Estimating pairwise covariance...')
    model_cov = CovarianceEst()
    Sigma_hat_lst, _, _, _ = model_cov.Train(M=200000,
                                             batch_size=20,
                                             lamb=100,
                                             X=torch.tensor(X_mnar_selfmasked, device=device).float()[:n, :],
                                             cond_lower=torch.tensor(list(cond_lower)),
                                             cond_upper=torch.tensor(list(cond_upper)),
                                             init_mu=None,
                                             init_var=None,
                                             affine_transform=True,
                                             display_per_iter=200)
    end = time()
    result_mu = mu_hat_lst.reshape(-1)
    Sigma_learned = LearnSigma(Sigma_hat_lst.detach().numpy())
    time_lst.append(end - start)

    print('Mean_ est:', result_mu, 'Mean true', mu)
    print("Covariance", Sigma_learned, 'Sigma true', Sigma)

    """
    Performance evaluation
    """
    mape_Sigma = np.mean(np.abs((Sigma - Sigma_learned.cpu().numpy()) / Sigma)) * 100
    mape_mu = np.mean(np.abs((mu - result_mu.cpu().numpy()) / Sigma)) * 100

    print("MAPE sigma", mape_Sigma)
    print("MAPE mu", mape_mu)

    mae_Sigma = np.mean(np.abs(Sigma - Sigma_learned.cpu().numpy()))
    mae_mu = np.mean(np.abs(mu - result_mu.cpu().numpy()))

    print("MAE sigma", mae_Sigma)
    print("MAE mu", mae_mu)

    rmse_Sigma = np.mean(np.abs(Sigma - Sigma_learned.cpu().numpy()) ** 2)
    rmse_mu = np.mean(np.abs(mu - result_mu.cpu().numpy()) ** 2)

    print("RMSE Sigma", rmse_Sigma)
    print("RMSE mu", rmse_mu)