#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import preprocessing
import config

p = config.setup()

def synthetic_data():
    N = p.n
    d = p.d
    mu = np.random.uniform(low=p.mu_low, high=p.mu_high, size=d)
    Sig_factor = np.random.uniform(low=p.sig_low, high=p.sig_high, size=(d, d))
    Sigma = Sig_factor.dot(Sig_factor.T)
    X = np.random.multivariate_normal(mu, Sigma, N)
    return X, N, d, mu, Sigma

def Airfoil_data():
    """
    UCI Airfoil Self-Noise data set: https://archive.ics.uci.edu/ml/datasets/airfoil+self-noise
    Number of instances: 1503
    Number of Attributes: 6
    Area: Physical
    Data donated: 2014-03-04

        Returns:
            X: Normalized data using min_max_scaler
            df_mu: Mean vector before normalization
            df_Sigma: Covariance matrix before normalization
            mu： mean vector after normalization
            Sigma: Covariance matrix after normalization

    Results for this dataset:

    kl divergence between target and estimation is:  5.497613087953148
    total variation distance based on Pinsker's inequality：  2.313420603317776
    """

    feature_names = ['Frequency', 'Angle', 'Chord Length', 'velocity', 'displacement thickness', 'sound pressure']
    df = pd.read_csv("airfoil_self_noise.csv", header=None, sep='\t', names=feature_names)

    df_mu = df.mean().values
    df_Sigma = df.cov().values
    N, d = df.shape

    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(df)
    df_normalized = pd.DataFrame(np_scaled)

    mu = df_normalized.mean().values
    Sigma = df_normalized.cov().values
    X = np.array(df_normalized.values)
    return X, N, d, mu, Sigma


def ALCOHOL_data():
    """
    UCI Alcohol QCM Sensor Dataset: https://archive.ics.uci.edu/ml/datasets/Alcohol+QCM+Sensor+Dataset
    Number of instances: 125
    Number of Attributes: 8
    Area: Computer
    Data donated: 2019-07-22

        Returns:
            X: Normalized data using min_max_scaler
            df_mu: Mean vector before normalization
            df_Sigma: Covariance matrix before normalization
            mu： mean vector after normalization
            Sigma: Covariance matrix after normalization

   kl divergence between target and estimation is:  92.42381248872101
   total variation distance based on Pinsker's inequality：  18.596715915303992
    """
    qcm3 = pd.read_csv('QCM3.csv', sep = ';')
    qcm6 = pd.read_csv('QCM6.csv', sep = ';')
    qcm7 = pd.read_csv('QCM7.csv', sep = ';')
    qcm10 = pd.read_csv('QCM10.csv', sep = ';')
    qcm12 = pd.read_csv('QCM12.csv', sep = ';')

    df = pd.concat([qcm3, qcm6, qcm7, qcm10, qcm12])
    df = df.drop(df.columns[[10, 11, 12, 13, 14]], axis=1)
    df_mu = df.mean().values
    df_Sigma = df.cov().values
    N, d = df.shape

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 2))
    np_scaled = min_max_scaler.fit_transform(df)
    df_normalized = pd.DataFrame(np_scaled)

    mu = df_normalized.mean().values
    Sigma = df_normalized.cov().values
    X = np.array(df_normalized.values)

    return X, N, d, mu, Sigma

def breast_cancer_data():
    """
    UCI ML Breast Cancer Wisconsin (Diagnostic): https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)
    Number of instances: 569
    Number of Attributes: 32
    Area: Life
    Data donated: 1995-11-01

        Returns:
            X: Normalized data using min_max_scaler
            df_mu: Mean vector before normalization
            df_Sigma: Covariance matrix before normalization
            mu： mean vector after normalization
            Sigma: Covariance matrix after normalization

    kl divergence between target and estimation is:  31.16952018040963
    total variation distance based on Pinsker's inequality：  6.419173451253687
    """
    from sklearn import datasets
    bc = datasets.load_breast_cancer()
    df = pd.DataFrame(bc.data, columns= bc.feature_names)

    df_mu = df.mean().values
    df_Sigma = df.cov().values

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 5))
    np_scaled = min_max_scaler.fit_transform(df)
    df_normalized = pd.DataFrame(np_scaled)

    mu = df_normalized.mean().values
    Sigma = df_normalized.cov().values
    N, d = df.shape
    X = np.array(df_normalized.values)
    return X, N, d, mu, Sigma

def wine_red_data():
    """
    Notes: Two datasets are included, related to red and white vinho verde wine samples, from the north of Portugal.
    UCI Wine Quality Data Set : https://archive.ics.uci.edu/ml/datasets/wine+quality
    Number of instances: 4898
    Number of Attributes: 12
    Area: Life
    Data donated: 2009-10-07

        Returns:
            X: Normalized data using min_max_scaler
            df_mu: Mean vector before normalization
            df_Sigma: Covariance matrix before normalization
            mu： mean vector after normalization
            Sigma: Covariance matrix after normalization

    kl divergence between target and estimation is:  8.750030127181608
    total variation distance based on Pinsker's inequality：  2.774052321654935
    """
    df = pd.read_csv('winequality-red.csv', low_memory=False, sep=';')
    df_mu = df.mean().values
    df_Sigma = df.cov().values

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 5))
    np_scaled = min_max_scaler.fit_transform(df)
    df_normalized = pd.DataFrame(np_scaled)

    mu = df_normalized.mean().values
    Sigma = df_normalized.cov().values
    X = np.array(df_normalized.values)
    N, d = df.shape
    return X, N, d, mu, Sigma

def wine_white_data():
    """
    kl divergence between target and estimation is:  4.214757963461256
    total variation distance based on Pinsker's inequality：  2.3667139683004743
    """
    df = pd.read_csv('winequality-white.csv', low_memory=False, sep=';')
    df_mu = df.mean().values
    df_Sigma = df.cov().values

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 5))
    np_scaled = min_max_scaler.fit_transform(df)
    df_normalized = pd.DataFrame(np_scaled)

    mu = df_normalized.mean().values
    Sigma = df_normalized.cov().values
    X = np.array(df_normalized.values)
    N, d = df.shape
    return X, N, d, mu, Sigma

def steel_industry_data():
    """
    UCI Steel Industry Energy Consumption Dataset Data Set:
    http://archive.ics.uci.edu/ml//datasets/Steel+Industry+Energy+Consumption+Dataset

    """
    df = pd.read_csv('Steel_industry_data.csv', low_memory=False, sep=',')
    df = df.drop(['date', 'WeekStatus', 'Day_of_week', 'Load_Type'], axis=1)
    df_mu = df.mean().values
    df_Sigma = df.cov().values

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 5))
    np_scaled = min_max_scaler.fit_transform(df)
    df_normalized = pd.DataFrame(np_scaled)

    mu = df_normalized.mean().values
    Sigma = df_normalized.cov().values
    X = np.array(df_normalized.values)
    N, d = df.shape
    return X, N, d, mu, Sigma


def happy_data():
    """
    Data from the World Happiness Report. For details see: https://worldhappiness.report/ed/2019/changing-world-happiness/
    Dataset GDP per capita is in terms of Purchasing Power Parity (PPP) adjusted to constant 2011 international dollars,
    taken from the World Development Indicators (WDI) released by the World Bank on November 14, 2018.
    The equation uses the natural log of GDP per capita, as this form fits the data significantly better than GDP per capita.

    Notes: https://colab.research.google.com/github/uu-sml/course-sml-public/blob/master/exercises/SML-session_9_features.ipynb#scrollTo=BwoEDXRROPpQ
    """
    df = pd.read_csv('https://uu-sml.github.io/course-sml-public/data/happy.csv', delimiter=';')

    df = df.drop(df.columns[[0, 1]], axis=1)
    df.head()
    df_mu = df.mean().values
    df_Sigma = df.cov().values

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 5))
    np_scaled = min_max_scaler.fit_transform(df)
    df_normalized = pd.DataFrame(np_scaled)

    mu = df_normalized.mean().values
    Sigma = df_normalized.cov().values
    X = np.array(df_normalized.values)
    N, d = df.shape
    return X, N, d, mu, Sigma

if __name__ == '__main__':
    X, N, d, mu, Sigma = synthetic_data()
    percentile = 0.675
    std = np.sqrt(np.diag(Sigma))
    cond_lower = mu + percentile * std
    cond_upper = [10000.0] * d
    for i in range(X.shape[1]):
        X[np.logical_and(X[:, i] > cond_lower[i], X[:, i] < cond_upper[i]), i] = np.nan