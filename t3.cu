#include<cuda.h>
#include<bits/stdc++.h>

#define number_of_nodes atoi(argv[1])
#define number_of_edges atoi(argv[2])
#define threads_per_block 256
#define number_of_blocks_nodes ((number_of_nodes/threads_per_block) + 1)
#define number_of_blocks_edges ((number_of_edges/threads_per_block) + 1)
#define pii std::pair<int,int>
#define KERNEL_CYCLES gpu_graph->V


struct Graph
{
    int V;  // number of vertices
    int excess_total;   // total excess flow over all vertices in the graph
    int *height;    // array containing height values of the vertices
    int *excess_flow;   // array containing excess flow values of the vertices
    pii *adj_mtx;    // array containing the adjacency matrix of the graph as (flow,capacity) pair edges
};

void preflow(Graph *cpu_graph, int source)
{
    for(int i = 0; i < cpu_graph->V; i++)
    {
        cpu_graph->height[i] = 0;
        cpu_graph->excess_flow[i] = 0;        
    }

    cpu_graph->height[source] = cpu_graph->V;

}

__global__ void push_relabel_kernel(Graph *gpu_graph)
{
    int cycle = KERNEL_CYCLES;
    unsigned int id = (blockIdx.x*blockDim.x) + threadIdx.x;
    int e1,e2,h1,h2;

    while(cycle > 0)
    {
        if(gpu_graph->excess_flow[id] > 0 && gpu_graph->height[id] < gpu_graph->V)
        {
            e1 = gpu_graph->excess_flow[id];
            h1 = INT_MAX;
        }

        for(int i = 0; i < gpu_graph->V; i++)
        {
            int ind = (gpu_graph->V*id) + i;

            if(gpu_graph->adj_mtx[ind].second - gpu_graph->adj_mtx[ind].first > 0)
            {
                h2 = gpu_graph->height[i];

                if(h2 < h1)
                {
                    // v1 = v;
                    h1 = h2;
                }
            }
        }

        if(gpu_graph->height[id] > h1)
        {
            int d = std::min(e1,gpu_graph->adj_mtx)
        }
    }
}

void global_relabel(Graph *cpu_graph);

void push_relabel(Graph *cpu_graph, Graph *gpu_graph, int source, int sink)
{
    while(cpu_graph->excess_flow[source] + cpu_graph->excess_flow[sink] < cpu_graph->excess_total)
    {
        cudaMemcpy(gpu_graph->height,cpu_graph->height,gpu_graph->V*sizeof(int),cudaMemcpyHostToDevice);
        
        push_relabel_kernel<<<number_of_blocks_nodes,threads_per_block>>>(gpu_graph);
        
        cudaMemcpy(cpu_graph->adj_mtx,gpu_graph->adj_mtx,cpu_graph->V*cpu_graph->V*sizeof(pii),cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_graph->height,gpu_graph->height,cpu_graph->V*sizeof(int),cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_graph->excess_flow,gpu_graph->excess_flow,cpu_graph->V*sizeof(int),cudaMemcpyDeviceToHost);
        
        global_relabel(cpu_graph);
    }
}


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

    // time start

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

    // print max_flow

    // time end

    // write times to file 

    // free device memory

    // free host memory
}