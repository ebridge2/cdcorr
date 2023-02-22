FROM neurodata/graspologic:latest

RUN pip install jupyterlab jupyter
RUN apt-get update
RUN apt-get install -y g++ gcc

RUN pip install seaborn dask dask[distributed]
RUN pip install git+https://github.com/ebridge2/dodiscover
RUN pip install git+https://github.com/neurodata/hyppo/@cdcorr
RUN pip install statsmodels
RUN apt-get install -y r-base
RUN pip install rpy2

RUN git clone --recursive-submodules git@github.com:ebridge2/cdcorr.git /cdcorr
RUN R -e "install.packages('GeneralisedCovarianceMeasure', dependencies=TRUE)"
