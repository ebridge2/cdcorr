# setwd("~/Documents/research/hypo-repos/cdcorr/cdcorr/simulations/")
suppressMessages(require(stats))
suppressMessages(require(weightedGCM))
suppressMessages(require(GeneralisedCovarianceMeasure))

fnames = list.files("./RCIT/R/", full.names=TRUE)
sapply(fnames, function(f) suppressMessages(source(f)))

cmanova <- function(Y, G, X) {
    Gf <- factor(G)
    mod_alt <- lm(Y ~ X + Gf + Gf:X)
    mod_null <- lm(Y ~ X)
    mod_verynull <- lm(Y ~ 1)
    test <- stats:::anova.mlmlist(mod_alt, mod_null)
    return(list(stat=test$Pillai[2], pvalue=test$`Pr(>F)`[2]))
}

kcit_wrap <- function(Y, G, X, nrep=1000) {
  res <- tryCatch({
    KCIT(Y, G, X, nrep=nrep)},
    error=function(e) {
      return(list(stat=NaN, pvalue=NaN))
    })
    return(list(stat=res$stat, pvalue=res$pvalue))
}

rcit_wrap <- function(Y, G, X, nrep=1000) {
    res <- tryCatch({
      RCIT(Y, G, X, approx="perm", nrep=nrep)},
      error=function(e) {
        return(list(stat=NaN, pvalue=NaN))
      })
    return(list(stat=res$stat, pvalue=res$pvalue))
}

rcot_wrap <- function(Y, G, X, nrep=1000) {
  res <- tryCatch({
    RCoT(Y, G, X, approx="perm", nrep=nrep)},
    error=function(e) {
      return(list(stat=NaN, pvalue=NaN))
    })
    return(list(stat=res$stat, pvalue=res$pvalue))
}

wgcm_wrap <- function(Y, G, X, nrep=1000) {
  res <- tryCatch({
    wgcm.est(Y, G, X, regr.meth="xgboost", nsim=nrep)},
    error=function(e) {
      return(list(stat=NaN, pvalue=NaN))
    })
  return(list(stat=NaN, pvalue=res))
}

gcm_wrap <- function(Y, G, X, nrep=1000) {
  res <- tryCatch({
    gcm.test(as.matrix(Y), as.matrix(G), as.matrix(X), regr.meth="xgboost", nsim=nrep)},
    error=function(e) {
      return(list(stat=NaN, pvalue=NaN))
    })
  return(list(stat=res$test.statistic, pvalue=res$p.value))
}