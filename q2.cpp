#include <mpi.h>
#include<bits/stdc++.h>

using namespace std;    

#define N 2000
#define MASTER 0   
#define MASTER_TASK 1
#define WORKER_TASK 2

double A[N][N], B[N][N], C[N][N], a[N][N], l[N][N], u[N][N];
int tasks, p, id, id2, workers, source, dest, mtype, rows, averow, extra, offset, rc;
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
    MPI_Comm_rank(MPI_COMM_WORLD, &id2);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    int mx_size = N;
    int tmp_size = mx_size -1;
    int diag_ref = 0; 
    std::cout<<"\n"<<"******* decomposition going on *******"<<"\n";
    if (id2>MASTER){
        for (int i=0; i<tmp_size; i++, diag_ref++){
        double *diag_row = a[diag_ref * mx_size + diag_ref];
        for (int j = diag_ref + 1; j < mx_size; j++) {
            if (j % p == id) {
            double *save = a[j * mx_size + diag_ref];
            forw_elim(&save, diag_row, mx_size - diag_ref);
            }
        }
        for (int j = diag_ref + 1; j < mx_size; j++) {
            double *save = a[j * mx_size + diag_ref];
            MPI_Bcast(save, mx_size - diag_ref, MPI_DOUBLE, j % p, MPI_COMM_WORLD);
        }       
    }
    }
    
    if (id2==MASTER){
        std::cout<<"\n"<<"matrix obtained after decomposition"<<"\n";
        for (int m=0; m<N; m++){
            for (int n=0; n<N; n++){
                if(n>=m){
                    std::cout<<*a[m * N + n]<<" ";
                }
                else{
                    cout<<0<<" ";
                }
            }
            std::cout<<"\n";
        }
        std::cout<<"\n"<<"task done!!"<<"\n";
    }
}

void mpi_matrix_multiplication(int argc, char** argv, double (&A)[N][N], double (&B)[N][N], int id){
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &tasks);
    if (tasks < 2 ) {
        printf("Need at least two MPI tasks. Quitting...\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
        exit(1);
    }
    workers = tasks-1; 

    if (id==MASTER){
        for (int i= 0; i< N; i++){
            for (int j= 0; j< N; j++){
                A[i][j] = rand() % (N);
                B[i][j] = rand() % (N);
                
            } 
        }
        std::cout<<"matrix A"<<"\n";
        for (int m=0; m<N; m++){
            for (int n=0; n<N; n++){
                std::cout<<A[m][n]<<" ";
            }
            std::cout<<"\n";
        }
        std::cout<<"\n"<<"matrix B"<<"\n";
        for (int m=0; m<N; m++){
            for (int n=0; n<N; n++){
                std::cout<<B[m][n]<<" ";
            }
            std::cout<<"\n";
        }
        std::cout<<"\n";
        averow = N/workers;
        extra = N%workers;
        offset = 0;
        mtype = MASTER_TASK;
        for (dest = 1; dest <= workers; dest++){
            rows = (dest <= extra) ? averow+1 : averow;
            MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&A[offset][0], rows*N, MPI_DOUBLE, dest, mtype,MPI_COMM_WORLD);
            MPI_Send(&B, N*N, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
            offset = offset + rows;
        }
        std::cout<<"message sent to worker for multiplication"<<"\n";

        mtype = WORKER_TASK;
        for (int i=1; i<=workers; i++){
            source = i;
            MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&C[offset][0], rows*N, MPI_DOUBLE, source, mtype, MPI_COMM_WORLD, &status);
        }
        std::cout<<"******* computation going on *******"<<"\n";
        std::cout<<"message received by master after multiplication"<<"\n";
        std::cout<<"\n"<<"matrix obtained after multiplication:"<<"\n";
        for (int m=0; m<N; m++){
            for (int n=0; n<N; n++){
                std::cout<<C[m][n]<<" ";
            }
            std::cout<<"\n";
        }
        std::cout<<"\n";
        std::cout<<"let's start matrix decomposition:"<<"\n";
        mpi_decomposition(A,l,u, N);
    }
    if (id > MASTER){
        mtype = MASTER_TASK;
        MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&A, rows*N, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&B, N*N, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
        // std::cout<<"message received by worker"<<"\n";
        for (int k=0; k<N; k++)
         for (int i=0; i<rows; i++)
         {
            C[i][k] = 0.0;
            for (int j=0; j<N; j++)
               C[i][k] += A[i][j] * B[j][k];
         }
        mtype = WORKER_TASK;
        MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&C, rows*N, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
        // std::cout<<"message sent to master after computation"<<"\n";
    }
    MPI_Finalize();
}

int main(int argc, char** argv){
    srand(1);
    mpi_matrix_multiplication(argc, argv, A, B, id);
}