

x1  <- rnorm(20)
x2  <- rnorm(20, mean=x1, sd=.01)
y  <-  rnorm(20, mean=3+x1+x2)

# ols model
ols_coef = lm(y~x1+x2)$coef
y_ols = ols_coef[1] + ols_coef[2] * x1 + ols_coef[3] * x2
y_ols
# ridge
library(MASS)
ridge_coef = lm.ridge(y~x1+x2, lambda=1)$coef
y_rid = ridge_coef[1] * x1 + ridge_coef[2] * x2




library(ggplot2)
df <- data.frame(x1,x2,y, y_ols, y_rid)

q <- qplot(x1, y_ols, data=df)
q + geom_line()


ggplot(df, aes(x1))+geom_line(aes(y=y_ols, colour="ols"))+geom_line(aes(y=y_rid, colour="ridge"))+geom_point(aes(x=x1, y=y))
q