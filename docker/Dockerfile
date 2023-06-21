# syntax=docker/dockerfile:1
FROM python:3.8-slim-buster
RUN pip3 install inquirer subprocess32 numpy scipy pandas scikit-learn brainspace pymanopt==0.2.5
RUN pip3 install h5py bctpy seaborn matplotlib
COPY micaopen/sf_prediction /opt/micaopen/sf_prediction
