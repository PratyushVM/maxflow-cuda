#include<cuda.h>
#include<bits/stdc++.h>

#define number_of_nodes 6
#define number_of_edges 10
#define threads_per_block 256
#define number_of_blocks_nodes ((number_of_nodes/threads_per_block) + 1)
#define number_of_blocks_edges ((number_of_edges/threads_per_block) + 1)
#define pii std::pair<int,int>
#define KERNEL_CYCLES gpu_graph->V
#define INF 1000000


struct Graph
{
    int V;  // number of vertices
    int excess_total;   // total excess flow over all vertices in the graph
    int *height;    // array containing height values of the vertices
    int *excess_flow;   // array containing excess flow values of the vertices
    pii *adj_mtx;    // array containing the adjacency matrix of the graph as (residual capacity,capacity) pair edges
};

int main(int argc, char **argv)
{
    // checking if sufficient arguments are passed in runtime
    if(argc < 5)
    {
        printf("Insufficient number of arguments\n");
        exit(0);
    }

    // reading the arguments passed in CLI
    int V = atoi(argv[1]);
    int E = atoi(argv[2]);
    int source = atoi(argv[3]);
    int sink = atoi(argv[4]);
    
    // initialising Graph variables for host and device
    Graph *cpu_graph,*gpu_graph;
    int *cpu_height,*gpu_height;
    int *cpu_excess_flow,*gpu_excess_flow;
    pii *cpu_adj_mtx,*gpu_adj_mtx;

    // allocating host memory for variables stored on the CPU
    cpu_graph = (Graph*)malloc(sizeof(Graph));
    cpu_height = (int*)malloc(V*sizeof(int));
    cpu_excess_flow = (int*)malloc(V*sizeof(int));
    cpu_adj_mtx = (pii*)malloc(V*V*sizeof(pii));

    // allocating CUDA device global memory for variables stored on the GPU
    cudaMalloc((void**)&gpu_graph,sizeof(Graph));
    cudaMalloc((void**)&gpu_height,V*sizeof(int));
    cudaMalloc((void**)&gpu_excess_flow,V*sizeof(int));
    cudaMalloc((void**)&gpu_adj_mtx,V*V*sizeof(pii));

    // Assigning values to the Graph object in the host memory
    cpu_graph->V = V;
    cpu_graph->excess_total = 0;
    cpu_graph->height = cpu_height;
    cpu_graph->excess_flow = cpu_excess_flow;
    cpu_graph->adj_mtx = cpu_adj_mtx;

    // readgraph() - add capacity values to cpu_adj_mtx
    readgraph(cpu_graph,V,E);

    // time start

    // invoking the preflow function 
    preflow(cpu_graph,source);

    // copying the graph from host memory to CUDA device global memory
    cudaMemcpy(gpu_graph,cpu_graph,sizeof(Graph),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_height,cpu_height,V*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_excess_flow,cpu_excess_flow,V*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_adj_mtx,cpu_adj_mtx,V*V*sizeof(pii),cudaMemcpyHostToDevice);

    // assigning values to pointers of the Graph object in the CUDA device global memory
    cudaMemcpy(&(gpu_graph->height),&gpu_height,sizeof(int*),cudaMemcpyHostToDevice);
    cudaMemcpy(&(gpu_graph->excess_flow),&gpu_excess_flow,sizeof(int*),cudaMemcpyHostToDevice);
    cudaMemcpy(&(gpu_graph->adj_mtx),&gpu_adj_mtx,sizeof(pii*),cudaMemcpyHostToDevice);

    // invoking the push_relabel host function
    push_relabel(cpu_graph,gpu_graph,source,sink);

    // copying the graph from the CUDA device global memory back to host memory
    cudaMemcpy(cpu_graph,gpu_graph,sizeof(Graph),cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_height,gpu_height,V*sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_excess_flow,gpu_excess_flow,V*sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_adj_mtx,gpu_adj_mtx,V*V*sizeof(pii),cudaMemcpyDeviceToHost);

    // assigning values to pointers of the Graph object in the host memory
    cpu_graph->height = cpu_height;
    cpu_graph->excess_flow = cpu_excess_flow;
    cpu_graph->adj_mtx = cpu_adj_mtx;

    // printing maximum flow of the flow network
    printf("The maximum flow of the flow network is %d\n",cpu_graph->excess_flow[sink]);

    // time end

    // write times to file 

    // free device memory

    // free host memory
}