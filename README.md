# maxflow-cuda
Implementation of the maximum network flow problem in CUDA. Uses the parallel push-relabel algorithm, as illustrated in this <a href = "https://www.sciencedirect.com/science/article/pii/B9780123859631000058"> paper</a> .

The program depicts the values of height, excess flow and the residual flow for each node at each iteration, along with the maximum flow obtained. It also runs a serial version of the push relabel algorithm and verifies the output.

## Execution :

```
To build the project, run :
make

```
```
To run the algorithm :
./maxflow <number of vertices> <number of edges> <source vertex> <sink vertex>

In file "edgelist.txt", edge input to be given in the form :
<end of edge 0> <end of edge 0> <capacity of edge 0>
<end of edge 1> <end of edge 1> <capacity of edge 1>
<end of edge 2> <end of edge 2> <capacity of edge 2>
...

```
```
To clean, run :
make clean
```

## Contents
1. Makefile
2. include - 
    <ul><li> serial_graph.h - for the serial implementation</li>
    <li> parallel_graph.cuh - for the parallel implementation</li>
    </ul>
3.src -
    <ul><li>graph_s.cpp - contains functions used by the serial check
    <li>io_par.cu - contains the input function and the print function used after each iteration
    <li>preflow.cu - contains the preflow/Init routine
    <li>push_relabel.cu - contains the host push relabel routine
    <li>push_relabel_kernel.cu - contains the kernel invoked as part of the push_relabel routine
    <li>global_relabel.cu - contains the heuristic host global relabel routine
    </ul>
## References

1. <a href = "https://www.sciencedirect.com/science/article/pii/B9780123859631000058">Efficient CUDA Algorithms for the Maximum Network Flow Problem</a>
2. http://www.cse.iitm.ac.in/~rupesh/teaching/gpu/jan20/
3. https://www.geeksforgeeks.org/push-relabel-algorithm-set-2-implementation/
4. Parallel implementation of flow and matching algorithms - Agnieszka Łupińska, Jagiellonian University, Kraków
5. https://www.geeksforgeeks.org/ford-fulkerson-algorithm-for-maximum-flow-problem/
