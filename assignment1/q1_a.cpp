#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

double f(double x) {
   double y;
   y = cos(x);
   return y;
}

double integral(double a, double b, int n, int thread_count) {
   double  h, approx;
   int  i;
   h = (b-a)/n; 
   approx = (f(a) + f(b))/2.0; 
#  pragma omp parallel for num_threads(thread_count) reduction(+: approx) 
   for (i = 1; i <= n-1; i++) {
     approx += f(a + i*h);
   }
   approx = h*approx; 
   return approx;
} 

int main(int argc, char* argv[]) {
   double global_result = 0.0;  // Store result in global_result 
   double a, b;                 // Left and right endpoints      
   int n;                       // Total number of trapezoids
   int thread_count;
   const double pi = 3.14159265358979323846;
   a = -pi/2;
   b = pi/2;
   n = 20000;
   thread_count = 32;
   global_result = integral(a, b, n, thread_count);

   printf("With n = %d trapezoids, our estimate ", n);
   printf("of the integral from -pi/2 to pi/2 = %f\n", global_result);
   return 0;
} 