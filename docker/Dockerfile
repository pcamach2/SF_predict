# syntax=docker/dockerfile:1
FROM python:3.8-slim-buster
RUN pip3 install cryptography==41.0.0 requests==2.31.0 certifi==2022.12.7
RUN pip3 install inquirer subprocess32 numpy scipy pandas scikit-learn brainspace pymanopt==0.2.5
RUN pip3 install h5py bctpy seaborn matplotlib
RUN pip3 install pymc3 statsmodels
COPY sf_prediction /opt/micaopen/sf_prediction
