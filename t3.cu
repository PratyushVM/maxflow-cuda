#include<cuda.h>
#include<bits/stdc++.h>

#define number_of_nodes atoi(argv[1])
#define number_of_edges atoi(argv[2])
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

void preflow(Graph *cpu_graph, int source)
{
    for(int i = 0; i < cpu_graph->V; i++)
    {
        cpu_graph->height[i] = 0;
        cpu_graph->excess_flow[i] = 0;        
    }

    cpu_graph->height[source] = cpu_graph->V;
    cpu_graph->excess_flow[source] = INF;
    cpu_graph->excess_total = INF;

}

__global__ void push_relabel_kernel(Graph *gpu_graph)
{
    int cycle = KERNEL_CYCLES;
    unsigned int u = (blockIdx.x*blockDim.x) + threadIdx.x;
    int e1,e2,h1,h2,v,v1,d;

    while(cycle > 0)
    {
        if(gpu_graph->excess_flow[u] > 0 && gpu_graph->height[u] < gpu_graph->V)
        {
            e1 = gpu_graph->excess_flow[u];
            h1 = INF;

            for(int i = 0; i < gpu_graph->V; i++)
            {
                int ind = (gpu_graph->V*u) + i;

                if(gpu_graph->adj_mtx[ind].second - gpu_graph->adj_mtx[ind].first > 0)
                {
                    v = i;
                    h2 = gpu_graph->height[i];

                    if(h2 < h1)
                    {
                        v1 = v;
                        h1 = h2;
                    }
                }
            }

            if(gpu_graph->height[u] > h1)
            {
                d = std::min(e1,(gpu_graph->adj_mtx[u*gpu_graph->V + v].first));
                atomicAdd((gpu_graph->adj_mtx[v1*gpu_graph->V + u].first), d);
                atomicSub((gpu_graph->adj_mtx[u*gpu_graph->V + v1].first), d);
                atomicAdd(gpu_graph->excess_flow[v1], d);
                atomicSub(gpu_graph->excess_flow[u], d);
            }
            else
            {
                gpu_graph->height[u] = h1 + 1;
            }
        }

        cycle = cycle - 1;

    }

}

void global_relabel(Graph *cpu_graph, int source, int sink)
{
    for(int u = 0; u < cpu_graph->V; u++)
    {
        for(int v = 0; v < cpu_graph->V; v++)
        {
            int ind = (u*cpu_graph->V) + v;
            int ind_trans = (v*cpu_graph->V) + u;

            if((cpu_graph->adj_mtx[ind].second - cpu_graph->adj_mtx[ind].first) > 0)
            {
                if(cpu_graph->height[u] > cpu_graph->height[v] + 1)
                {
                    cpu_graph->excess_flow[u] -= (cpu_graph->adj_mtx[ind].first);
                    cpu_graph->excess_flow[v] += (cpu_graph->adj_mtx[ind].first);
                    cpu_graph->adj_mtx[ind_trans].first += (cpu_graph->adj_mtx[ind].first);
                    cpu_graph->adj_mtx[ind] = 0; 
                }

            }

        }

        bool mark[cpu_graph->V];
        memset(mark,false,sizeof(mark));

        // bfs routine
        std::list<int> queue;
        int x = source;
        int level = cpu_graph->V;

        mark[source] = true;
        queue.push_back(source);

        while(!queue.empty())
        {
            x = queue.front();
            cpu_graph->height[x] = level;
            queue.pop_front();

            for(int i = 0; i < cpu_graph->V; i++)
            {
                if(cpu_graph->adj_mtx[x*cpu_graph->V + i].f > 0 && !mark[i])
                {
                    mark[i] = true;
                    cpu_graph->height[i] = level - 1;
                }
            }
            level -= 1;
        }

        for(int i = 0; i < cpu_graph->V; i++)
        {
            if(mark[i] == false)
            {
                mark[i] = true;
                cpu_graph->excess_total -= cpu_graph->excess_flow[i];
            }
        }

    }
    
}

void push_relabel(Graph *cpu_graph, Graph *gpu_graph, int source, int sink)
{
    while(cpu_graph->excess_flow[source] + cpu_graph->excess_flow[sink] < cpu_graph->excess_total)
    {
        cudaMemcpy(gpu_graph->height,cpu_graph->height,gpu_graph->V*sizeof(int),cudaMemcpyHostToDevice);
        
        push_relabel_kernel<<<number_of_blocks_nodes,threads_per_block>>>(gpu_graph);
        
        cudaMemcpy(cpu_graph->adj_mtx,gpu_graph->adj_mtx,cpu_graph->V*cpu_graph->V*sizeof(pii),cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_graph->height,gpu_graph->height,cpu_graph->V*sizeof(int),cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_graph->excess_flow,gpu_graph->excess_flow,cpu_graph->V*sizeof(int),cudaMemcpyDeviceToHost);
        
        global_relabel(cpu_graph,source,sink);
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
    printf("The maximum flow of the flow network is %d\n",cpu_graph->excess_flow[sink]);

    // time end

    // write times to file 

    // free device memory

    // free host memory
}