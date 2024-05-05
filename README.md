# MPI
Matrix Multiplication Using MPI on IBISCO Cluster: 

Introduction: 
Previously during past times, all the computations were performed on a single processor. So single processor has to perform all the necessary calculations such as arithmetic operations. This process is time-consuming. To overcome this problem and to utilize the powerful available resources parallel computing has been introduced. This is a special type of computing technology where multiple processors/cores perform the computation by working together. As a result, a big complex task is evenly distributed and load balanced. It reduces the pressure on a single processor. In this process, the entire workload is subdivided into multiple smaller tasks and distributed among several processors which are available in the parallel computing resources. This significantly reduces the computation time and enables large-scale computation

Problem Statement: 
One particular algorithm has to be implemented which should be executed on the IBISCO Cluster using MPI. So, this work aims to compute a matrix multiplication among two square matrices and the result of this multiplication will be stored in another square matrix. The task should be divided into multiple different processes. If we consider n=number of processes, then we will take different values of n starting from 1 to some 16 and compare the computation time for different numbers of processes. We will consider two larger matrices of N=700 so that the problem would be complex and well distributed among different processes. If two small metrics are considered then the complexity of the problem won’t be suitable for the parallel computing can be performed using traditional architecture only.

Implementation WorkFlow:

![image](https://github.com/semanto-mondal/MPI/assets/133217806/20bf261f-d7d7-442e-bcd5-c2f5c07a98bf)

Result Analysis: 

As the number of processes increases the computation time decreases significantly. From the above plot, we can visualize that till nprocs is equal to 14 with the increase of nprocs, computation time reduces whereas after 14 with the increase of nprocs the computation time starts increasing. This phenomenon is called Scaling Inefficiency. It is a common problem in parallel computing that after a threshold point with the increase of nprocs the computation time increases instead of decreasing. One cause which leads to this problem is communication overhead. As the number of processors used in the computation is increased, the amount of communication required between processors can also increase. This communication can involve sending and receiving data between processors, synchronizing the computation across processors, and managing the distribution of work among processors. 
![image](https://github.com/semanto-mondal/MPI/assets/133217806/71c7b1ee-ba07-40b0-bada-ddd0d086eb96)
