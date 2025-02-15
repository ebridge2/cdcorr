FROM neurodata/graspologic:latest

RUN pip install jupyterlab jupyter
RUN apt-get update
RUN apt-get install -y g++ gcc
RUN apt-get install -y python3-venv
RUN apt-get install -y r-base

RUN python -m venv /root/.virtualenvs/causal

RUN /root/.virtualenvs/causal/bin/pip install \
    seaborn dask dask[distributed] \
    git+https://github.com/ebridge2/dodiscover \
    statsmodels rpy2

RUN R -e "install.packages('GeneralisedCovarianceMeasure', dependencies=TRUE)"
RUN R -e "install.packages('weightedGCM', dependencies=TRUE)"
RUN R -e "install.packages(c('devtools', 'energy', 'vegan', 'cdcsis', 'reticulate'), dependencies=TRUE)"
RUN R -e "library(devtools); install_github('neurodata/causal_batch')"
