#' tableprob.cpp with C++ to Calculate the Exact Probability per Table
#' 
#' tableprob.cpp with C++ to calculate the exact probability per table, which runs much faster than table.prob() with R codes.
#' @examples
#' library(Rcpp)
#' library(RcppArmadillo)
#' Rcpp::sourceCpp("tableprob.cpp")
#'
#' #Probability to select the A-arm based on utinity function of E.st: E.st_utinity()
#' E.st_prob.A<-apply(ABSF,1,E.st_utinity)
#' E.table.probability.v<-table.prob(E.st_prob.A,N=100,pA=0.8,pB=0.6)
#' E.table.probability.v
#'
#' #Probability to select the A-arm based on utinity function of T.st: T.st_utinity() 
#' T.st_prob.A<-apply(ABSF,1,T.st_utinity,w=0.5)
#' T.table.probability.v<-tableprob(T.st_prob.A,N=100,pA=0.8,pB=0.6)
#' T.table.probability.v


// LinearRegression.cpp
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace arma;  // use the Armadillo library for matrix computations
using namespace Rcpp;

#include <iostream>
#include <fstream>
#include <iomanip>
using namespace std;

#include <string.h>
#include <stdlib.h>
#include <math.h>

// [[Rcpp::export]]
vec tableprob(vec decexp,int N,double pA,double pB){
  int i,j,k,l;
  int n;
  
  double p[2];
  p[0] = pA;
  p[1] = pB;
  
  if(p[0] < 0 || p[0] > 1.0 || p[1] < 0 || p[1] > 1.0){
    cout << "PA and PB must be between 0 and 1." << endl;
    return decexp;
  }
  
  struct onelist{
    double **m;
  };
  onelist **s1,**ret;
  
  int slcnt;
  slcnt = 0;
  for(i=0;i<=N;i++){
    for(j=0;j<=i;j++){
      for(l=0;l<i+1-j;l++){
        for(k=0;k<j+1;k++){
          slcnt++;
        }
      }
    }
  }
  if(slcnt != (int)decexp.size()){
    cout << slcnt << "\t" << (int)decexp.size() << endl;
    cout << "N does not correspond to size of vector." << endl;
    return decexp;
  }
  
  slcnt = 0;
  s1 = new onelist*[N+1];
  ret = new onelist*[N+1];
  for(i=0;i<=N;i++){
    s1[i] = new onelist[i+1];
    ret[i] = new onelist[i+1];
    for(j=0;j<=i;j++){
      //cout << "List:" << i+1 << "\tSub list:" << j+1 << endl;
      s1[i][j].m = new double*[j+1];
      ret[i][j].m = new double*[j+1];
      for(k=0;k<j+1;k++){
        s1[i][j].m[k] = new double[i+1-j];
        ret[i][j].m[k] = new double[i+1-j];
      }
      
      for(l=0;l<i+1-j;l++){
        for(k=0;k<j+1;k++){	  
          s1[i][j].m[k][l] = (double)decexp[slcnt];
          slcnt++;
          ret[i][j].m[k][l] = s1[i][j].m[k][l];
        }
      }
      
    }
  }  
  
  
  ret[0][0].m[0][0] = 1.0;
  
  double **tmpselect;
  double dims[2];
  int J;
  for(i=0;i<N;i++){
    //cout << "N:" << i+1 << endl;
    n = i+1;
    for(j=0;j<=n;j++){
      for(k=0;k<j+1;k++){
        for(l=0;l<n+1-j;l++){
          ret[n][j].m[k][l] = 0.0;
        }
      }
    }
    
    for(j=0;j<=i;j++){
      tmpselect = s1[i][j].m;
      dims[0] = j+1;
      dims[1] = i+1-j;
      J = j+1;     
      for(k=0;k<dims[0];k++){
        for(l=0;l<dims[1];l++){
          ret[n][J].m[k][l] += tmpselect[k][l]*ret[i][j].m[k][l]*p[0];
        }
      }
      for(k=0;k<dims[0];k++){
        for(l=0;l<dims[1];l++){
          ret[n][J].m[k+1][l] += tmpselect[k][l]*ret[i][j].m[k][l]*(1-p[0]);
        }
      }
      J=j;
      for(k=0;k<dims[0];k++){
        for(l=0;l<dims[1];l++){
          ret[n][J].m[k][l] += (1-tmpselect[k][l])*ret[i][j].m[k][l]*p[1];
        }
      }
      for(k=0;k<dims[0];k++){
        for(l=0;l<dims[1];l++){
          ret[n][J].m[k][l+1] += (1-tmpselect[k][l])*ret[i][j].m[k][l]*(1-p[1]);
        }
      }          
    } 
  }
  
  double sump;
  for(i=0;i<=N;i++){
    sump = 0.0;
    for(j=0;j<=i;j++){
      for(k=0;k<j+1;k++){
        for(l=0;l<i+1-j;l++){
          sump += ret[i][j].m[k][l];
        }
      }	 
    }
  }
  
  vec retexp = decexp;
  slcnt = 0;
  for(i=0;i<=N;i++){
    for(j=0;j<=i;j++){
      for(l=0;l<i+1-j;l++){
        for(k=0;k<j+1;k++){
          retexp[slcnt] = ret[i][j].m[k][l];
          slcnt++;
        }
      }
    }
  }
  
  for(i=0;i<=N;i++){
    for(j=0;j<=i;j++){
      for(k=0;k<j+1;k++){
        delete[] s1[i][j].m[k];
        delete[] ret[i][j].m[k];
      }
      delete[] s1[i][j].m;
      delete[] ret[i][j].m;
    }
    delete[] s1[i];
    delete[] ret[i];
  }
  delete[] s1;
  delete[] ret;
  
  return retexp;
}