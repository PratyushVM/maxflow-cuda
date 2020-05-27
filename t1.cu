#include<cuda.h>
#include<bits/stdc++.h>

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
    std::vector<Edge> edgelist; // edges for which the Vertex is the source node

    // constructor function of a Vertex
    Vertex(int h, int e){

        this->height = h;
        this->excess_flow = e;
    }

};

class Graph{

    int V;  // total number of nodes/vertices
    std::vector<Vertex> adj_list;    // vector containing the Vertex list
    int excess_total;       // total excess flow of all active vertices (including source and sink)

    void preflow(int source);
    __global__ void preflow_init(int source);
    __global__ void preflow_kernel(int source);

    void push_relabel();
    // insert push_relabel_kernels as necessary
    
    void global_relabel();

public:

    // constructor function for Graph object
    Graph(int v){

        this->V = v;
        this->excess_total = 0;

    }

    void addEdge(int s, int d, int f, int c);

    int maximum_flow(int source, int sink);

};

Graph::addEdge(int source, int destination, int flow, int capacity){

    this->adj_list[source].edgelist.push_back(Edge(destination,flow,capacity));

};

Graph::preflow(int source){

    preflow_init<<<number_of_blocks,threads_per_block>>>(source);
    // init excess flow Vertex
    // init height Vertex
    // perform preflow 

}

Graph::__global__ void preflow_init(int source){

    unsigned int id = (blockIdx.x*blockDim.x) + threadIdx.x;
    
    if(id < V){
     
        adj_list[id].height = 0;
        adj_list[id].excess_flow = 0;

        if(id == source){

            preflow_kernel<<<adj_list[source].edgelist.size()/threads_per_block + 1,threads_per_block>>>(source);

        }

    }

}

Graph::__global__ void preflow_kernel(int source){

    unsigned int id = (blockIdx.x*blockDim.x) + threadIdx.x;

    if(id < adj_list[source].edgelist.size()){

        adj_list[source].edgelist[id].flow = adj_list[source].edgelist[id].capacity;
        // change to atomics!!!
        adj_list[adj_list[source].edgelist[id].destination].excess_flow += adj_list[source].edgelist[id].flow; 
        adj_list[adj_list[source].edgelist[id].destination].edgelist.push_back(Edge(source,-adj_list[source].edgelist[id].flow,0));
    
    }
    
}




int main(int argc, char **argv[]){

    Graph cpu_graph(atoi(argv[1]));
    // readgraph function, argument in cli - number of nodes, rest in a txt file maybe ?

    
    
    
    return 0;
}




