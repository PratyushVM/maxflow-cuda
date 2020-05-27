#include<cuda.h>
#include<bits/stdc++.h>
#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
#include<thrust/copy.h>

#define number_of_threads atoi(argv[1])
#define threads_per_block 256
#define number_of_blocks ((number_of_threads/threads_per_block) + 1)


struct Edge{

    int flow;   // flow of edge
    int capacity;   // capacity of edge
    int destination;    // destination node of edge from the Vertex

    // constructor function of an Edge
    Edge(int d, int f, int c){

        this->destination = d;
        this->flow = f;
        this->capacity = c;
    
    }

};

struct Vertex{

    int height; // height of the Vertex
    int excess_flow;    // excess flow of the Vertex
    thrust::host_vector<Edge> edgelist; // edges for which the Vertex is the source node

    // constructor function of a Vertex
    Vertex(int h, int e){

        this->height = h;
        this->excess_flow = e;
    }

};

class Graph{

    int V;  // total number of nodes/vertices
    thrust::host_vector<Vertex> adj_list;    // vector containing the Vertex list
    int excess_total;       // total excess flow of all active vertices (including source and sink)

public: 

    void addEdge(int s, int d, int f, int c);

};

struct Vertex_gpu{

    int height; // height of the Vertex
    int excess_flow;    // excess flow of the Vertex
    thrust::device_vector<Edge> edgelist; // edges for which the Vertex is the source node

    // constructor function of a Vertex
    Vertex(int h, int e){

        this->height = h;
        this->excess_flow = e;
    }

};

class Graph_gpu{

    int V;  // total number of nodes/vertices
    thrust::device_vector<Vertex_gpu> adj_list;    // vector containing the Vertex list
    int excess_total;       // total excess flow of all active vertices (including source and sink)

public: 

    void addEdge(int s, int d, int f, int c);

};


// Function prototypes

__global__ void preflow_init(Graph* g, int source);
__global__ void preflow_kernel(Graph* g, int source);
void push_relabel();
// insert push_relabel_kernels as necessary
void global_relabel();
int maximum_flow(int source, int sink);


// Function codes

Graph::addEdge(int source, int destination, int flow, int capacity){

    this->adj_list[source].edgelist.push_back(Edge(destination,flow,capacity));

};

Graph_gpu::addEdge(int source, int destination, int flow, int capacity){

    this->adj_list[source].edgelist.push_back(Edge(destination,flow,capacity));

};


__global__ void preflow_init(Graph* g, int source){

    unsigned int id = (blockIdx.x*blockDim.x) + threadIdx.x;
    
    if(id < g->V){
     
        g->adj_list[id].height = 0;
        g->adj_list[id].excess_flow = 0;

        if(id == source){

            g->adj_list[id].height = g->V;
            preflow_kernel<<<g->adj_list[source].edgelist.size()/threads_per_block + 1,threads_per_block>>>(source);

        }

    }

}

Graph::__global__ void preflow_kernel(Graph* g, int source){

    unsigned int id = (blockIdx.x*blockDim.x) + threadIdx.x;

    if(id < g->adj_list[source].edgelist.size()){

        g->adj_list[source].edgelist[id].flow = g->adj_list[source].edgelist[id].capacity;
        // change to atomics!!!  --- Changed, verify if it works
        atomicAdd(&(g->adj_list[adj_list[source].edgelist[id].destination].excess_flow), g->adj_list[source].edgelist[id].flow);

        atomicAdd(&(g->excess_total),g->adj_list[source].edgelist[id].flow); 
        
        g->adj_list[adj_list[source].edgelist[id].destination].edgelist.push_back(Edge(source,-g->adj_list[source].edgelist[id].flow,0));
    
    }
    
}




int main(int argc, char **argv[]){

    if(argc < 4){

        printf("Not enough arguments\n");
        exit(0);
    }

    Graph *cpu_graph;
    Graph_gpu *gpu_graph;
    
    cpu_graph->V = atoi(argv[1]);
    cpu_graph->excess_total = 0;

    int source,sink;
    source = atoi(argv[2]);
    sink = atoi(argv[3]);

    // readgraph function, argument in cli - number of nodes, source, sink, and rest in a txt file maybe ?
    // pass cpu_graph to readgraph 
    // may add graph generator function later

    // Allocate device memory for gpu_graph -- WILL THIS WORK???
    cudaMalloc((void**)&gpu_graph,sizeof(*cpu_graph));

    // copy cpu_graph to gpu_graph on device memory -- WILL THIS WORK???
    cudaMemcpy(gpu_graph->V,cpu_graph->V,sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_graph->excess_total,cpu_graph->excess_total,sizeof(int),cudaMemcpyHostToDevice);
    // !!! NO IDEA IF THIS WORKS...
    gpu_graph->adj_list = cpu_graph->adj_list;

    // perform initialization and preflow routine in parallel
    preflow_init<<<number_of_blocks,threads_per_block>>>(gpu_graph,source);

    // push relabel invoked from host, (the while routine)

    // copy data back from device to host

    // print max flow
    
    return 0;

}






