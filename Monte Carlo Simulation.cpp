#' @examples

#' library(devtools)
#' library(Rcpp)
#' library(RcppArmadillo)
#' Rcpp::sourceCpp("simdecision.1.cpp")

#' N <- 500
#' n.iter <- 10000
#' ps.<-combn(seq(0,1,by=0.01),2)
#' mean.success.exp<-matrix(0,N+1,length(ps.[1,]))
#' for(i in seq_along(ps.[1,])){
#'   outs.tmp<- matrix(0,n.iter,N+1)
#'   for(j in 1:n.iter){
#'     frac.success<-expdecision2x2(n=N,ps=ps.[,i])
#'     outs.tmp[j,]<-frac.success[,1]
#'   }
#'   ABSR<-apply(outs.tmp,2,mean)
#'   mean.success.exp[,i]<-ABSR
#' }
#' (mean.success.exp,file="mean.success.exp.sim.500.RData")

#Flexible target strategy
# N <-500
# n.iter<-10000
# # ps.<-combn(seq(0,1,by=0.01),2)
# weights.0.0005<-c(seq(-1,-0.06,by=0.01),seq(-0.05,0.05,by=0.0005),seq(0.06,1,by=0.01))
# mean.success.wpbeta<-array(0,c(length(weights.0.0005),length(ps.[1,]),N+1))
#mean.success.wpbeta<-array(0,c(length(weights.0.0005),length(ps.[1,]),N+1))
#for(i in seq_along(weights.0.0005)){
# print(i)
#  for(j in seq_along(ps.[1,])){
#    outs.tmp<- matrix(0,n.iter,N+1)
#    for(k in 1:n.iter){
#    frac.success<-fledecision2x2(n=N,ps=ps.[,j],target=weights.0.0005[i])
#   outs.tmp[k,]<-frac.success[,1]
#  }
#  ABSR<-apply(outs.tmp,2,mean)
#  mean.success.wpbeta[i,j,]<-ABSR
#  }
#    }
# save(mean.success.wpbeta,file="mean.success.wpbeta.sim.500.RData")




// LinearRegression.cpp
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace arma;  // use the Armadillo library for matrix computations
using namespace Rcpp;


// [[Rcpp::export]]
double probexp(double x, double y, double z, double w){
  double ret =0;
  double pa = (x+1)/(x+y+2);
  double pb = (z+1)/(z+w+2);
  if(pa==pb){
    ret = 0.5;
  }else if(pa>pb){
    ret = 1;
  }
  return ret;
}

// [[Rcpp::export]]
double probtargetFlex(double x, double y, double z, double w,double k=0){
  double ret =0;
  double expa = (x+1)/(x+y+2);
  double expb = (z+1)/(z+w+2);
  double t = std::max(expa,expb);double tm;
  if(k >= 0){
    tm = t + (1-t)*k;
  }else{
    tm = t +k*t;
  }
  double pa = Rf_pbeta(tm,x+1,y+1,0,0);
  double pb = Rf_pbeta(tm,z+1,w+1,0,0);
  if(pa==pb){
    ret = 0.5;
  }else if(pa>pb){
    ret = 1;
  }
  return ret;
}

// [[Rcpp::export]]
vec expdecision2x2(int n, vec ps) {
  mat M = mat(n+1,4);
  vec frac(n+1);
  M.zeros();
  ivec A = randi(n, distr_param(0,1));
  vec R0 = randu(n);
  vec R = randu(n);
  for(int i=0; i<n;i++){
    double q = 0.5;
    q = probexp(M(i,0),M(i,1),M(i,2),M(i,3));
    //double q = 0.5;
    int a = 1;
    if(R0[i] < q){
      a = 0;
    }
    int tmp = 1;
    if(R[i] < ps[a]){
      tmp = 0;
    }
    tmp = a*2+tmp;
    M.row(i+1) = M.row(i);
    M(i+1,tmp)++;
  }
  
  for(int i=0;i<=n;i++)
    frac(i) = (M(i,0)+M(i,2))/(M(i,0)+M(i,1)+M(i,2)+M(i,3));
  frac(0)=0.5;
  return frac;
}




// [[Rcpp::export]]
vec fledecision2x2(int n, vec ps,double target) {
  mat M = mat(n+1,4);
  vec frac(n+1);
  M.zeros();
  ivec A = randi(n, distr_param(0,1));
  vec R0 = randu(n);
  vec R = randu(n);
  for(int i=0; i<n;i++){
    double q = 0.5;
    q = probtargetFlex(M(i,0),M(i,1),M(i,2),M(i,3),target);
    
    //double q = 0.5;
    int a = 1;
    if(R0[i] < q){
      a = 0;
    }
    int tmp = 1;
    if(R[i] < ps[a]){
      tmp = 0;
    }
    tmp = a*2+tmp;
    M.row(i+1) = M.row(i);
    M(i+1,tmp)++;
  }
  
  for(int i=0;i<=n;i++)
    frac(i) = (M(i,0)+M(i,2))/(M(i,0)+M(i,1)+M(i,2)+M(i,3));
  frac(0)=0.5;
  return frac;
}



// END