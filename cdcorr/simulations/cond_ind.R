# setwd("~/Documents/research/hypo-repos/cdcorr/cdcorr/simulations/")
require(stats)

fnames = list.files("./RCIT/R/", full.names=TRUE)
sapply(fnames, function(f) source(f))

cmanova <- function(Y, G, X) {
    Gf <- factor(G)
    mod_alt <- lm(Y ~ X + Gf + Gf:X)
    mod_null <- lm(Y ~ X)
    mod_verynull <- lm(Y ~ 1)
    test <- stats:::anova.mlmlist(mod_alt, mod_null)
    return(list(stat=test$Pillai[2], pvalue=test$`Pr(>F)`[2]))
}

kcit_wrap <- function(Y, G, X, nrep=1000) {
    res <- KCIT(Y, G, X, nrep=nrep)
    return(list(stat=res$stat, pvalue=res$pvalue))
}

rcit_wrap <- function(Y, G, X, nrep=1000) {
    res <- RCIT(Y, G, X, approx="perm", nrep=nrep)
    return(list(stat=res$stat, pvalue=res$pvalue))
}

rcot_wrap <- function(Y, G, X, nrep=1000) {
    res <- RCoT(Y, G, X, approx="perm", nrep=nrep)
    return(list(stat=res$stat, pvalue=res$pvalue))
}