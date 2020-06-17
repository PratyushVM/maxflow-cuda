#include<cuda.h>
#include<bits/stdc++.h>
// include thrust to use in readgraph

#define number_of_nodes 6
#define number_of_edges 10
#define threads_per_block 256
#define number_of_blocks_nodes ((number_of_nodes/threads_per_block) + 1)
#define number_of_blocks_edges ((number_of_edges/threads_per_block) + 1)
#define KERNEL_CYCLES gpu_graph->V
#define INF 1000000

struct Edge
{
    int to; // destination node of the edge
    int flow;   // residual flow value of the edge
    int capacity;   // capacity of the edge
}

struct Graph
{
    int V;  // number of vertices
    int excess_total;   // total excess flow across all active nodes
    int *height;    // array containing height values of the vertices
    int *excess_flow;   // array containing excess flow values of the vertices
    Edge* edgelist; // array of edges of the graph
    int *index;  // array containing pairs of indices of edges in edgelist array corresponding to the nodes
}

int main(int argc, char **argv)
{
    // checking if sufficient number of arguments are passed in CLI
    if(argc < 5)
    {
        printf("Insufficient number of arguments passed during execution\n");
        exit(0);
    }

    // reading the arguments passed in CLI
    int V = atoi(argv[1]);
    int E = atoi(argv[2]);
    int source = atoi(argv[3]);
    int sink = atoi(argv[4]);

    // initialising graph variables for host and device
    Graph *cpu_graph,*gpu_graph;
    int *cpu_height,*gpu_height;
    int *cpu_excess_flow,*gpu_excess_flow;
    Edge *cpu_edgelist,*gpu_edgelist;
    int *cpu_index,*gpu_index;

    // allocating host memory for variables stored on the CPU
    cpu_graph = (Graph*)malloc(sizeof(Graph));
    cpu_height = (int*)malloc(V*sizeof(int));
    cpu_excess_flow = (int*)malloc(V*sizeof(int));
    cpu_edgelist = (Edge*)malloc(2*E*sizeof(Edge));
    cpu_index = (int*)malloc(V*sizeof(int) + 1);

    // allocating CUDA device global memory for variables stored on the GPU
    cudaMalloc((void**)&gpu_graph,sizeof(Graph));
    cudaMalloc((void**)&gpu_height,V*sizeof(int));
    cudaMalloc((void**)&gpu_excess_flow,V*sizeof(int));
    cudaMalloc((void**)&gpu_edgelist,2*E*sizeof(Edge));
    cudaMalloc((void**)&gpu_index,V*sizeof(int) + 1);

    // Assigning values to the Graph object in the host memory
    cpu_graph->V = V;
    cpu_graph->excess_total = 0;
    cpu_graph->height = cpu_height;
    cpu_graph->excess_flow = cpu_excess_flow;
    cpu_graph->edgelist = cpu_edgelist;
    cpu_graph->index = cpu_index;

    // add readgraph function to get edgelist,index from txt file - dont forget to add rev edges for each edge added
    // ...

    // time start

    // preflow fn()

    // push relabel host function - invokes pr kernel - global relbl

    // copy to host

    // time end

    // print result

    // free host var

    // free dev var
    
}
