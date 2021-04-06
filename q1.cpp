#include <omp.h>
#include<bits/stdc++.h>
using namespace std;

#define N 1000

double A[N][N], B[N][N], C[N][N], a[N][N], l[N][N], u[N][N];

void decomposition(double (&a)[N][N], double (&l)[N][N], double (&u)[N][N], int size){
    #pragma omp parallel shared(a,l,u)
    for (int i=0; i<size; i++){
        #pragma omp for schedule(static)
        for (int j=0; j<size; j++){
            if(j<i){
                l[j][i]=0;
                continue;
            }
            l[j][i] = a[j][i];
            for (int k=0; k<i; k++){
                l[j][i] = l[j][i] - l[j][k] * u[k][i];
            }
        }
        #pragma omp for schedule(static)
        for (int j = 0; j < size; j++)
        {
            if (j < i)
            {
                u[i][j] = 0;
                continue;
            }
            if (j == i)
            {
                u[i][j] = 1;
                continue;
            }
            u[i][j] = a[i][j] / l[i][i];
            for (int k = 0; k < i; k++)
            {
                u[i][j] = u[i][j] - ((l[i][k] * u[k][j]) / l[i][i]);
            }
        
        }
    }
}

int main() 
{
    int i,j,k;
    srand(1);
    omp_set_num_threads(omp_get_num_procs());
    for (i= 0; i< N; i++){
        for (j= 0; j< N; j++){
            A[i][j] = rand() % (N);
            B[i][j] = rand() % (N);
	    } 
    }  
    #pragma omp parallel for private(i,j,k) shared(A,B,C)
    for (i = 0; i < N; ++i) {
        for (j = 0; j < N; ++j) {
            for (k = 0; k < N; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    // for (int m=0; m<N; m++){
    //     for (int n=0; n<N; n++){
    //         std::cout<<C[m][n]<<" ";
    //     }
    //     std::cout<<"\n";
    // }
    std::cout<<"\n";
    std::cout<<"let's decompose the matrix now:"<<"\n";
    decomposition(A,l,u, N);
    // for (int m=0; m<N; m++){
    //     for (int n=0; n<N; n++){
    //         std::cout<<u[m][n]<<" ";
    //     }
    //     std::cout<<"\n";
    // }
}