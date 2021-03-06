#-------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
# 
#   http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
#-------------------------------------------------------------

# JUnit test class: dml.test.integration.descriptivestats.UnivariateStatsTest.java
# command line invocation assuming $S_HOME is set to the home of the R script
# Rscript $S_HOME/Scale.R $S_HOME/in/ $S_HOME/expected/
args <- commandArgs(TRUE)
options(digits=22)

library("psych")
library("moments")
library("Matrix")

V = readMM(paste(args[1], "vector.mtx", sep=""))
P = readMM(paste(args[1], "prob.mtx", sep=""))

n = nrow(V)

# mean
mu = mean(V)

# variances
var = var(V[,1])

# standard deviations
std_dev = sd(V[,1], na.rm = FALSE)

# standard errors of mean
SE = sd(V[,1])/sqrt(sum(!is.na(V[,1])))

# coefficients of variation
cv = std_dev/mu

# harmonic means (note: may generate out of memory for large sparse matrices becauses of NaNs)
# har_mu = harmonic.mean(V[,1]) -- DML does not support this yet

# geometric means is not currently supported.
# geom_mu = geometric.mean(V[,1]) -- DML does not support this yet

# min and max
mn=min(V)
mx=max(V)

# range
rng = mx - mn

# Skewness
g1 = moment(V[,1], order=3, central=TRUE)/(std_dev^3)

# standard error of skewness (not sure how it is defined without the weight)
se_g1=sqrt( 6*n*(n-1.0) / ((n-2.0)*(n+1.0)*(n+3.0)) )

# Kurtosis (using binomial formula)
g2 = moment(V[,1], order=4, central=TRUE)/(var^2)-3

# Standard error of Kurtosis (not sure how it is defined without the weight)
se_g2= sqrt( (4*(n^2-1)*se_g1^2)/((n+5)*(n-3)) )

# median
md = median(V[,1]) #quantile(V[,1], 0.5, type = 1)

# quantile
Q = t(quantile(V[,1], P[,1], type = 1))

# inter-quartile mean
S=c(sort(V[,1]))

q25d=n*0.25
q75d=n*0.75
q25i=ceiling(q25d)
q75i=ceiling(q75d)

iqm = sum(S[(q25i+1):q75i])
iqm = iqm + (q25i-q25d)*S[q25i] - (q75i-q75d)*S[q75i]
iqm = iqm/(n*0.5)

#print(paste("IQM ", iqm));

out_minus = t(as.numeric(V< mu-5*std_dev)*V) 
out_plus = t(as.numeric(V> mu+5*std_dev)*V)

write(mu, paste(args[2], "mean", sep=""));
write(std_dev, paste(args[2], "std", sep=""));
write(SE, paste(args[2], "se", sep=""));
write(var, paste(args[2], "var", sep=""));
write(cv, paste(args[2], "cv", sep=""));
# write(har_mu),paste(args[2], "har", sep=""));
# write(geom_mu, paste(args[2], "geom", sep=""));
write(mn, paste(args[2], "min", sep=""));
write(mx, paste(args[2], "max", sep=""));
write(rng, paste(args[2], "rng", sep=""));
write(g1, paste(args[2], "g1", sep=""));
write(se_g1, paste(args[2], "se_g1", sep=""));
write(g2, paste(args[2], "g2", sep=""));
write(se_g2, paste(args[2], "se_g2", sep=""));
write(md, paste(args[2], "median", sep=""));
write(iqm, paste(args[2], "iqm", sep=""));
writeMM(as(t(out_minus),"CsparseMatrix"), paste(args[2], "out_minus", sep=""), format="text");
writeMM(as(t(out_plus),"CsparseMatrix"), paste(args[2], "out_plus", sep=""), format="text");
writeMM(as(t(Q),"CsparseMatrix"), paste(args[2], "quantile", sep=""), format="text");
