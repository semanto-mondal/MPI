#include <stdio.h>
#include <mpi.h>

#define N 700

int main(int argc, char **argv) {
    int my_rank, num_procs;
    float A[N][N] = {{0}};
    float B[N][N] = {{0}};
    float C[N][N] = {0};
    float C_sum[N][N] = {0}; // New buffer to hold the sum of C from each process
    int i, j, k;
    double start_time, end_time, computation_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    start_time=MPI_Wtime();
    // Initialize A and B
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            A[i][j] = (i + j + 1.0f) / 100.0f;
            B[i][j] = (i * j + 1.0f) / 10000.0f;
        }
    }

    // Divide the work among the processes
    int rows_per_proc = N / num_procs;
    int start_row = my_rank * rows_per_proc;
    int end_row = start_row + rows_per_proc;
    if (my_rank == num_procs - 1 && end_row < N) {
        end_row = N;
    }

    // Compute the matrix multiplication
    for (i = start_row; i < end_row; i++) {
        for (j = 0; j < N; j++) {
            for (k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    // Combine the results
    MPI_Reduce(C, C_sum, N*N, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Copy the sum of C from each process back to C on process 0
    if (my_rank == 0) {
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                C[i][j] = C_sum[i][j];
            }
        }
    }

    end_time=MPI_Wtime();

    // Print the results
    if (my_rank == 0) {
        printf("Matrix A:\n");
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                printf("%f ", A[i][j]);
            }
            printf("\n");
        }

        printf("\nMatrix B:\n");
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                printf("%f ", B[i][j]);
            }
            printf("\n");
        }

        printf("\nMatrix C:\n");
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                printf("%f ", C[i][j]);
            }
            printf("\n");

        }
       computation_time=(end_time - start_time)*1000;
       printf("\nComputation time: %f milliseconds\n", computation_time);


    }
    MPI_Finalize();

    return 0;
}