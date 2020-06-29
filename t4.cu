#include<cuda.h>
#include<bits/stdc++.h>
// include thrust to use in readgraph

#define number_of_nodes 6
#define number_of_edges 10
#define threads_per_block 256
#define number_of_blocks_nodes ((number_of_nodes/threads_per_block) + 1)
#define number_of_blocks_edges ((number_of_edges/threads_per_block) + 1)
#define INF 1000000

#define IDX(i,x,y) (i)*(x) + (y)

struct Edge
{
    int from;   // source node of the edge
    int to; // destination node of the edge
    int rflow;   // residual flow value of the edge
    int capacity;   // capacity of the edge
};

bool cmp(const Edge &x, const Edge &y)
{
    return (x.from > y.from);
}

struct Graph
{
    int V,E;  // number of vertices and number of edges
    int excess_total;   // total excess flow across all active nodes
    int *height;    // array containing height values of the vertices
    int *excess_flow;   // array containing excess flow values of the vertices
    Edge *edgelist; // array of edges of the graph
    int *index;  // array containing pairs of indices of edges in edgelist array corresponding to the nodes
};

void readgraph(int V, int E, int source, int sink, Graph *cpu_graph)
{
    FILE *fp = fopen("edgelist.txt","r");
    
    char buf1[10],buf2[10],buf3[10];
    int e1,e2,cp;

    /*
    for(int i = 0; i < 2*E; i++)
    {
        cpu_graph->edgelist[i].rflow = 0;
        cpu_graph->edgelist[i].capacity = 0;
    }
    */

    Edge input[2*E];
    for(int i = 0; i < 2*E; i+=2)
    {
        fscanf(fp,"%s",buf1);
        fscanf(fp,"%s",buf2);
        fscanf(fp,"%s",buf3);

        e1 = atoi(buf1);
        e2 = atoi(buf2);
        cp = atoi(buf3);

        input[i].rflow = 0;
        input[i].capacity = cp;
        input[i].from = e1;
        input[i].to = e2;

        input[i+1].rflow = 0;
        input[i+1].capacity = 0;
        input[i+1].from = e2;
        input[i+1].to = e1;
        
    }   
    
    std::sort(input,input+2*E,cmp);

    memcpy(input,cpu_graph->edgelist,2*E*sizeof(Edge));

    cpu_graph->index[0] = 0;
    int ind_v = 1;
    for(int i = 1; i < 2*cpu_graph->E; i++)
    {
        if(cpu_graph->edgelist[i].from != cpu_graph->edgelist[i-1].from)
        {
            cpu_graph->index[ind_v] = i;
            ind_v++;
        }
    }

}


void preflow(int V, int E, Graph *cpu_graph, int *cpu_height, int *cpu_excess_flow, Edge *cpu_edgelist, int *cpu_index, int source)
{
    for(int i = 0; i < V; i++)
    {
        cpu_height[i] = 0;
        cpu_excess_flow[i] = 0;        
    }

    cpu_height[source] = V;
    cpu_excess_flow[source] = 0;

    for(int i = 0; i < 2*E; i++)
    {
        // for all x,y
        cpu_edgelist[i].rflow = cpu_edgelist[i].capacity;
    }

    for(int i = cpu_index[source]; i < cpu_index[source + 1]; i++)
    {
        // for s,x
        cpu_edgelist[i].rflow = 0;
        
        // for x,s
        for(int j = cpu_index[cpu_edgelist[i].to]; j < cpu_index[cpu_edgelist[i].to + 1]; j++)
        {
            if(cpu_edgelist[j].to == source)
            {
                cpu_edgelist[j].rflow = cpu_edgelist[j].capacity + cpu_edgelist[i].capacity;
            }
        }
        
        cpu_excess_flow[cpu_edgelist[i].to] = cpu_edgelist[i].capacity; 
        cpu_graph->excess_total += cpu_edgelist[i].capacity;
    
    }
    
}

void global_relabel_cpu(int source, int sink,Graph *cpu_graph, int *cpu_height, int *cpu_excess_flow, Edge *cpu_edgelist, int *cpu_index, int *marking)
{    
    for(int i = 0; i < 2*cpu_graph->E; i++)
    {
        int x = cpu_edgelist[i].from;
        int y = cpu_edgelist[i].to;
        int rev_index;

        if(cpu_height[x] > (cpu_height[y] + 1) )
        {
            cpu_excess_flow[x] -= cpu_edgelist[i].rflow;
            cpu_excess_flow[y] += cpu_edgelist[i].rflow;

            // capture index of y-x
            for(int j = cpu_index[y]; j < cpu_index[y + 1]; j++)
            {
                if(cpu_edgelist[j].to == x)
                {
                    rev_index = j;
                    break;
                }
            }

            cpu_edgelist[rev_index].rflow += cpu_edgelist[i].rflow;
            cpu_edgelist[i].rflow = 0;

        }
                
    }

    // backward bfs
    int level = cpu_graph->V;
    std::list<int> queue;
    
    bool visit[cpu_graph->V];
    memset(visit,false,sizeof(visit));

    queue.push_back(source);
    cpu_height[source] = level;
    marking[source] = 1;
    visit[source] = true;

    int p,i;

    while(!queue.empty())
    {
        p = queue.front();
        queue.pop_front();

        for( i = cpu_index[i]; i < cpu_index[i + 1]; i++)
        {
            int q = cpu_edgelist[i].to;
            
            if((visit[q] == false) && (marking[q] != 2) )
            {
                queue.push_back(q);
                visit[q] = true;
                marking[q] = 1;
                cpu_height[q] = cpu_height[p] - 1;
            }

        }

    }

    for( i = 0; i < cpu_graph->V; i++)
    {
        if(marking[i] == 1)
        {
            marking[i] = 0;
        }
        if(marking[i] == 0)
        {
            marking[i] = 2;
            cpu_graph->excess_total -= cpu_excess_flow[i];
        }
    }

}


__global__ void push_relabel_kernel(Graph *gpu_graph, int *gpu_height, Edge *gpu_edgelist, int *gpu_excess_flow, int *gpu_index)
{
    // xth node is operated by xth thread
    unsigned int x = (blockIdx.x*blockDim.x) + threadIdx.x;
    if(x < gpu_graph->V)
    {
        int CYCLE = gpu_graph->V;

        int e_dash,y,y_dash,h_dash,h_double_dash,delta,index_1,index_2;

        while(CYCLE > 0)
        {
            if( (gpu_excess_flow[x] > 0) && (gpu_height[x] < gpu_graph->V) )
            {
                e_dash = gpu_excess_flow[x];
                y_dash = NULL;
                h_double_dash = INF;

                for(int i = gpu_index[x]; i < gpu_index[x + 1]; i++)
                {
                    y = gpu_edgelist[i].to;

                    h_dash = gpu_height[y];

                    if(h_double_dash > h_dash)
                    {
                        h_double_dash = h_dash;
                        y_dash = y;
                    }
                }



                if(h_double_dash < gpu_height[x])
                {
                    // capture cf x ydash
                    for(int i = gpu_index[x]; i < gpu_index[x + 1]; i++)
                    {
                        if(gpu_edgelist[i].to == y_dash)
                        {
                            index_1 = i;
                            break;
                        }
                    }

                    // capture cf ydash x
                    for(int i = gpu_index[y_dash]; i < gpu_index[y_dash + 1]; i++)
                    {
                        if(gpu_edgelist[i].to == x)
                        {
                            index_2 = i;
                            break;
                        }
                    }

                    // push towards y
                    delta = e_dash;
                    atomicMin(&delta,gpu_edgelist[index_1].rflow);
                    
                    atomicSub(&(gpu_edgelist[index_1].rflow),delta);
                    atomicAdd(&(gpu_edgelist[index_2].rflow),delta);
                    atomicSub(&(gpu_excess_flow[x]),delta);
                    atomicAdd(&(gpu_excess_flow[y_dash]),delta);

                }
                
                else
                {
                    // perform relabel
                    gpu_height[x] = h_double_dash + 1;
                }

            }

            CYCLE = CYCLE - 1;

        }
    
    }
}


void push_relabel(int V, int E, Graph *cpu_graph, Graph *gpu_graph, int *cpu_height, int *gpu_height, Edge *cpu_edgelist, Edge *gpu_edgelist, int *cpu_excess_flow, int *gpu_excess_flow, int *cpu_index, int *gpu_index, int source, int sink)
{
    while(cpu_graph->excess_flow[source] + cpu_graph->excess_flow[sink] < cpu_graph->excess_total)
    {
        cudaMemcpy(gpu_height,cpu_height,V*sizeof(int),cudaMemcpyHostToDevice);
        
        push_relabel_kernel<<<number_of_blocks_nodes,threads_per_block>>>(gpu_graph,gpu_height,gpu_edgelist,gpu_excess_flow,gpu_index);
        
        cudaMemcpy(cpu_edgelist,gpu_edgelist,2*E*sizeof(Edge),cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_height,gpu_height,V*sizeof(int),cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_excess_flow,gpu_excess_flow,V*sizeof(int),cudaMemcpyDeviceToHost);
        
        int *marking;
        marking = (int*)malloc(V*sizeof(int));
        memset(marking,0,sizeof(marking));

        global_relabel_cpu(source,sink,cpu_graph,cpu_height,cpu_excess_flow,cpu_edgelist,cpu_index,marking);
    
    }

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
    cpu_graph->E = E;
    cpu_graph->excess_total = 0;
    cpu_graph->height = cpu_height;
    cpu_graph->excess_flow = cpu_excess_flow;
    cpu_graph->edgelist = cpu_edgelist;
    cpu_graph->index = cpu_index;

    // add readgraph function to get edgelist,index from txt file - !!dont forget to add rev edges for each edge added!!
    //readgraph(int V, int E, int source, int sink, Graph *cpu_graph);

    // time start

    // preflow fn()
    preflow(V,E,cpu_graph,cpu_height,cpu_excess_flow,cpu_edgelist,cpu_index,source);

    // copy graph to device
    cudaMemcpy(gpu_graph,cpu_graph,sizeof(Graph),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_height,cpu_height,V*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_excess_flow,cpu_excess_flow,V*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_edgelist,cpu_edgelist,2*E*sizeof(Edge),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_index,cpu_index,V*sizeof(int) + 1,cudaMemcpyHostToDevice);

    cudaMemcpy(&(gpu_graph->height),&gpu_height,sizeof(int*),cudaMemcpyHostToDevice);
    cudaMemcpy(&(gpu_graph->excess_flow),&gpu_excess_flow,sizeof(int*),cudaMemcpyHostToDevice);
    cudaMemcpy(&(gpu_graph->edgelist),&gpu_edgelist,sizeof(Edge*),cudaMemcpyHostToDevice);
    cudaMemcpy(&(gpu_graph->index),&gpu_index,sizeof(int*),cudaMemcpyHostToDevice);

    // push relabel host function - invokes pr kernel - global relbl
    push_relabel(V,E,cpu_graph,gpu_graph,cpu_height,gpu_height,cpu_edgelist,gpu_edgelist,cpu_excess_flow,gpu_excess_flow,cpu_index,gpu_index,source,sink);

    // time end

    // print result
    printf("The maximum flow of the network is %d\n",cpu_excess_flow[sink]);
    
    // free host var

    // free dev var
    
}
