#include<cuda.h>
#include<bits/stdc++.h>

#define number_of_nodes 4
#define number_of_edges 3
#define threads_per_block 256
#define number_of_blocks_nodes ((number_of_nodes/threads_per_block) + 1)
#define number_of_blocks_edges ((number_of_edges/threads_per_block) + 1)
#define INF 1000000


void readgraph()
{

}

__global__ void push_relabel_kernel()
{

}

void push_relabel()
{

}

void global_relabel()
{

}

int main(int argc, char **argv)
{
    // checking if sufficient number of arguments are passed in CLI
    if(argc != 5)
    {
        printf("Invalid number of arguments passed during execution\n");
        exit(0);
    }

    // reading the arguments passed in CLI
    int V = atoi(argv[1]);
    int E = atoi(argv[2]);
    int source = atoi(argv[3]);
    int sink = atoi(argv[4]);

    // declaring variables to store graph data on host as well as on CUDA device global memory 
    int *cpu_height,*gpu_height;
    int *cpu_excess_flow,*gpu_excess_flow;
    int *Excess_total;
//    int *cpu_edgelist,*gpu_edgelist;
//    int *cpu_rflow,*gpu_rflow;
//    int *cpu_index,*gpu_index;
    
    // allocating host memory
    
    // allocating CUDA device global memory

    // readgraph

    // preflow - init

    // copy to device

    // push_relabel()

    // print value

    // free host memory

    // free device memory


    return 0;

}
