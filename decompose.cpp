#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <string>
#include<iostream>
#include<random>

using namespace std;

#define N 4
#define MASTER 0   
#define MASTER_TASK 1
#define WORKER_TASK 2

double A[N][N], B[N][N], C[N][N], a[N][N], l[N][N], u[N][N];
int tasks, tasks2, id, id2, workers, source, dest, mtype, rows, averow, extra, offset, rc;
MPI_Status status;


void forw_elim(double **origin, double *master_row, size_t dim)
{
    if (**origin == 0)
        return;

    double k = **origin / master_row[0];

    int i;
    for (i = 1; i < dim; i++) {
        (*origin)[i] = (*origin)[i] - k * master_row[i];
    }
    **origin = k;
}

void mpi_decomposition(double (&a)[N][N], double (&l)[N][N], double (&u)[N][N], int size){
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &id2);
    MPI_Comm_size(MPI_COMM_WORLD, &tasks2);
    int mx_size = tasks2;
    int tmp_size = mx_size -1;
    int diag_ref = 0;
    workers = tasks-1; 

    if (id2>MASTER){
        for (int i=0; i<tmp_size; i++, diag_ref++){
        double *diag_row = a[diag_ref * mx_size + diag_ref];
        for (int j = diag_ref + 1; j < mx_size; j++) {
            if (j % tasks2 == id) {
            double *save = a[j * mx_size + diag_ref];
            forw_elim(&save, diag_row, mx_size - diag_ref);
            }
        }
        for (int j = diag_ref + 1; j < mx_size; j++) {
            double *save = a[j * mx_size + diag_ref];
            MPI_Bcast(save, mx_size - diag_ref, MPI_DOUBLE, j % tasks2, MPI_COMM_WORLD);
        }       
    }
    }
    if (id2==MASTER){
        std::cout<<"bello";
        for (int m=0; m<N; m++){
            for (int n=0; n<N; n++){
                std::cout<<a[m][n]<<" ";
            }
            std::cout<<"\n";
        }
    }
    MPI_Finalize();
}