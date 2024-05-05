# MPI
Matrix Multiplication Using MPI on IBISCO Cluster: 

**Introduction:** 
Previously during past times, all the computations were performed on a single processor. So single processor has to perform all the necessary calculations such as arithmetic operations. This process is time-consuming. To overcome this problem and to utilize the powerful available resources parallel computing has been introduced. This is a special type of computing technology where multiple processors/cores perform the computation by working together. As a result, a big complex task is evenly distributed and load balanced. It reduces the pressure on a single processor. In this process, the entire workload is subdivided into multiple smaller tasks and distributed among several processors that are available in the parallel computing resources. This significantly reduces the computation time and enables large-scale computation

**Problem Statement:**
One particular algorithm has to be implemented which should be executed on the IBISCO Cluster using MPI. So, this work aims to compute a matrix multiplication among two square matrices and the result of this multiplication will be stored in another square matrix. The task should be divided into multiple different processes. If we consider n=number of processes, then we will take different values of n starting from 1 to some 16 and compare the computation time for different numbers of processes. We will consider two larger matrices of N=700 so that the problem would be complex and well distributed among different processes. If two small metrics are considered then the complexity of the problem won’t be suitable for the parallel computing can be performed using traditional architecture only.

**MPI FUNCTIONS:**

Different MPI FUNCTIONS are used to perform parallel computation seamlessly. A few common macros and functions are described as follows:

1.	MPI_Init: It initializes the MPI environment as well as bridges communication between the processes. 
2.	MPI_COMM_WORLD: It indicates all the processes in the program and it is a global communicator. 
3.	MPI_SUM: It computes the sum of input values and it’s a reduction operation. It is commonly used to perform the summation of the intermediate computation computed by individual processes. 
4.	MPI_FLOAT: It is used to define the precession in the floating point number.
5.	MPI_Send: It is used to send messages to other processes. 
6.	MPI_Recv: It is used to receive messages from other processes.
7.	MPI_Comm_rank: This function retrieves the current rank of the MPI Processes. 
8.	MPI_Comm_size:  This function retrieves the total number of processes in the MPI_COMM_WORLD.
9.	MPI_Wtime: It retrieves the elapsed time from the starting of the epoch in seconds.
10.	MPI_Reduce: It performs the reduction operation. It combines the results of several processes and gives a single result. 
11.	MPI_Finalize: This function shuts down the MPI environment at the end of the program and frees any resources allocated by MPI_Init.


**Implementation WorkFlow:**

![image](https://github.com/semanto-mondal/MPI/assets/133217806/20bf261f-d7d7-442e-bcd5-c2f5c07a98bf)

**Step-by-Step Implementation:**


### Overview

This document explains the process of parallel matrix multiplication using MPI (Message Passing Interface) with code examples.

### Steps

1. **Importing necessary libraries:**

    ```c
    #include <stdio.h>
    #include <mpi.h>
    ```

    It imports the MPI library which is used for parallel processing.

2. **Defining the size of the Matix and Main Function:**

    ```c
    #define N 700
    int main(int argc, char **argv){
    ```

    It defines the size of the square matrices as a 700*700 dimensional matrix and the main function.

3. **Declaration of Variables:**

    ```c
    int my_rank, num_procs;
    float A[N][N] = {{0}};
    float B[N][N] = {{0}};
    float C[N][N] = {0};
    float C_sum[N][N] = {0};
    int i, j, k;
    double start_time, end_time, computation_time;
    ```

    This code declares variables to be used later in the code, including the rank and number of processes, the three matrices A, B, and C, and variables for indexing and timing.

4. **MPI Initialization:** 

    ```c
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    start_time=MPI_Wtime();
    ```

    In this section, MPI is initialized and the rank as well as the number of processes for the current process are also defined. It also initializes a timer to measure computation time.

5. **Initialization of Matrices A and B:**

    ```c
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            A[i][j] = (i + j + 1.0f) / 100.0f;
            B[i][j] = (i * j + 1.0f) / 10000.0f;
        }
    }
    ```

    In this section, the values of Matrix A and B are assigned. Type of  A and B is floating point number and values are assigned based on the indices of the square matrix where N is set to 700.

6. **Dividing the work among multiple processes:**

    ```c
    int rows_per_proc = N / num_procs;
    int start_row = my_rank * rows_per_proc;
    int end_row = start_row + rows_per_proc;
    if (my_rank == num_procs - 1 && end_row < N) {
        end_row = N;
    }
    ```

    In general, the tasks are not equally distributed. For this reason, when we use a different number of processes each time we might get different answers for the same computation. To overcome this problem
   it is necessary to equally divide the workload among all the processes. This code divides the work of matrix multiplication among the processes by computing the number of rows each process is responsible   
   for and the starting and ending index of those rows. If the last process has fewer rows to compute than the others, it will take care of the remaining rows.

7. **Computing Matrix Multiplication:**
```c
for (i = start_row; i < end_row; i++) {
    for (j = 0; j < N; j++) {
        for (k = 0; k < N; k++) {
            C[i][j] += A[i][k] * B[k][j];
        }
    }
}
```
This code computes the matrix multiplication for the local row block of matrices A and B and stores the result in matrix C.

8. **Combining the result:**
```c
MPI_Reduce(C, C_sum, N*N, MPI_FLOAT, MPI_SUM, 0,   
MPI_COMM_WORLD);
```
The entire process is divided into several local and global arithmetic operations. Each Process performs some local operation after that all the outcomes of several local operations are combined into a global operation. This code combines the results of matrix multiplication from all processes using the MPI_Reduce function and stores the sum of each element in C_sum on process 0.

9. **Copy the sum of C from each process back to C on process 0:**
```c
if (my_rank == 0) {
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            C[i][j] = C_sum[i][j];
        }
    }
}

```
Here my_rank==0 means the master process. All the intermediate summations are stored in buffer C_sum. After the computation is completed MPI_Reduce stores the result to C_sum. However the final result should be in the original matrix C. The nested for-loops iterate over each row and column of C and copy the corresponding value from C_sum into C to ensure that the final result is stored successfully in the original matrix. 

10. **Ending the timer and printing the results**
```c
end_time=MPI_Wtime();
if (my_rank == 0) {
    printf("Matrix A:\n");
    // Print matrix A
    printf("\nMatrix B:\n");
    // Print matrix B
    printf("\nMatrix C:\n");
    // Print matrix C
    computation_time=(end_time - start_time)*1000;
    printf("\nComputation time: %f milliseconds\n",			computation_time);
}

```
If the current rank is 0 that is if the process is the master process then the timer is ended and the computation time is calculated in milliseconds. Finally prints the output.

11. **Ending the timer and printing the results**
```c
MPI_Finalize();
}

```
This code finalizes MPI ends the program and releases all the resources that were in use.



12. **Shell Script**
This is an executable file that is used to compile and execute the cpi.c program on a cluster using the Slurm job scheduler.

i.	Setting the environment variables:
```c
export UCX_NET_DEVICES=mlx5_0:1
export UCX_IB_GPU_DIRECT_RDMA=yes
export UCX_TLS=ib
```
UCX_NET_DEVICES variable specifies the network device to use, UCX_IB_GPU_DIRECT_RDMA enables GPU direct RDMA (enables high-speed communication between GPU and CPU), and UCX_TLS sets the transport layer to use InfiniBand.

ii.	Checking the command line arguments:
```c
if [ $# -ne 1 ]
then
echo "Usage: $0 <NPROCS>"
exit 1
fi
```
This code checks whether or not the script was called with only one argument which is the number of processes. Otherwise, it will generate an error.

iii.	Compiling the program:
```c
echo "Compile program"
make
echo
```
Here make is used to call the Makefile. It is used to compile and build executable programs. It also prints “Compile program” which indicates that the program is being compiled.

iv.	Execute the program:
```c
echo "Execute program"
srun -N $1 -p parallel --reservation=maintenance ./cpi
echo
```
This is to use the Slurm Scheduler to perform the execution of the program.

v.	Clean up temporary files:
```c
echo "Clear temporary file"
make clean
echo
```
This code prints a message to the console and runs make clean to remove temporary files generated during compilation.




13. **Makefile**
```c
CC=mpicc // sets the mpicc compiler

all: cpi //This line specifies that all target depends on the cpi target

cpi: cpi.o
        $(CC) -o cpi cpi.o //generate the object file and make cpi executable

clean:
        rm cpi cpi.o //clean up generated files


```
The last step is to make the Makefile which defines which compiler to use and also simplifies the build process including the object file.

**Result Analysis:**

As the number of processes increases the computation time decreases significantly. From the above plot, we can visualize that till nprocs is equal to 14 with the increase of nprocs, computation time reduces whereas after 14 with the increase of nprocs the computation time starts increasing. This phenomenon is called Scaling Inefficiency. It is a common problem in parallel computing that after a threshold point with the increase of nprocs the computation time increases instead of decreasing. One cause that leads to this problem is communication overhead. As the number of processors used in the computation increases, the amount of communication required between processors can also increase. This communication can involve sending and receiving data between processors, synchronizing the computation across processors, and managing the distribution of work among processors. 
![image](https://github.com/semanto-mondal/MPI/assets/133217806/71c7b1ee-ba07-40b0-bada-ddd0d086eb96)
