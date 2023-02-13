# setwd("~/Documents/research/hypo-repos/cdcorr/cdcorr/simulations/")
require(stats)

cmanova <- function(Y, G, X) {
  Gf <- factor(G)
  mod_alt <- lm(Y ~ X + Gf + Gf:X)
  mod_null <- lm(Y ~ X)
  mod_verynull <- lm(Y ~ 1)
  test <- stats:::anova.mlmlist(mod_alt, mod_null)
  return(list(stat=test$Pillai[2], pvalue=test$`Pr(>F)`[2]))
}
# 
# Yn <- as.matrix(read.csv('./data/sigsim_Y_null.csv', header=FALSE, sep= " "))
# Xn <- as.matrix(read.csv('./data/sigsim_X_null.csv', header=FALSE, sep= " "))
# Gn <- factor(as.matrix(read.csv('./data/sigsim_T_null.csv', header=FALSE, sep= " ")))
# # 
# Y <- as.matrix(read.csv('./data/sigsim_Y_effect.csv', header=FALSE, sep= " "))
# X <- as.matrix(read.csv('./data/sigsim_X_effect.csv', header=FALSE, sep= " "))
# G <- as.matrix(read.csv('./data/sigsim_T_effect.csv', header=FALSE, sep= " "))
# # 
# cmanova(Yn, Gn, Xn)
# cmanova(Y, G, X)

