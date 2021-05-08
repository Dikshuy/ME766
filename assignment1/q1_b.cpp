#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include<time.h>
#include<cmath>
#include<iostream>
#include<random>

static long iterations = 1000000000;
int thread_count = 2;

double montecarlo(){
  const double pi = 3.14159265358979323846;
  double a = -pi/2;
  double b = pi/2;
  double ySum = 0.0;
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(a,b);
  #  pragma omp parallel for num_threads(thread_count) reduction(+: ySum)
  for (int i=0; i<iterations; ++i) {
    double x = distribution(generator);
    // std::cout<<x<<" ";
    double y = cos(x);
    ySum += y;
  }
  double yAverage = ySum / double(iterations);
  double width = b-a;
  double height = yAverage;

  return width*height;
}

int main(){
  double val = montecarlo();
  std::cout<< val<<"\n";
  return 0;
}