#include "mpi.h"
#include<bits/stdc++.h>

using namespace std;

#define N 5
#define MASTER 0   

int main(int argc, char** argv){
    int tasks, id, workers, source, dest, mtype, rows, averow, extra, offset, rc;
    double A[N][N], B[N][N], C[N][N];
    srand(1);
    MPI_Status status;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &tasks);

    if (tasks < 2 ) {
        printf("Need at least two MPI tasks. Quitting...\n");
        exit(1);
    }

    workers = tasks - 1;
    if (id == 0){
        for (int i= 0; i< N; i++){
            for (int j= 0; j< N; j++){
                A[i][j] = rand() % (N);
                B[i][j] = rand() % (N);
            } 
        } 
        averow = N/workers;
        extra = N%workers;
        offset = 0;
        mtype = 1;
        for (dest = 1; dest <= workers; dest++){
            rows = (dest <= extra) ? averow+1 : averow;
            MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&A[offset][0], rows*N, MPI_DOUBLE, dest, mtype,MPI_COMM_WORLD);
            MPI_Send(&B, N*N, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
            offset = offset + rows;
        }
        mtype = 2;
        for (int i=1; i<=workers; i++){
            source = i;
            MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&C[offset][0], rows*N, MPI_DOUBLE, source, mtype, MPI_COMM_WORLD, &status);
        }
    }
    if (id > 0){
        mtype = 1;
        MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&A, rows*N, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&B, N*N, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);

        for(int i=0; i<N; i++){
            for (int j=0; j<N; j++){
                C[j][i] = 0.0;
                for(int k=0; k<N; k++){
                    C[j][i] += A[j][k]+B[k][i];
                }
            }
        }
        mtype = 2;
        MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&C, rows*N, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    for (int m=0; m<N; m++){
        for (int n=0; n<N; n++){
            std::cout<<C[m][n]<<" ";
        }
        std::cout<<"\n";
    } 
    std::cout<<"came here multiple times"<<"\n";
    return 0;
}

