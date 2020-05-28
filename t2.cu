#include<cuda.h>
#include<bits/stdc++.h>

#define number_of_nodes atoi(argv[1])
#define number_of_edges atoi(argv[2])
#define threads_per_block 256
#define number_of_blocks_nodes ((number_of_nodes/threads_per_block) + 1)
#define number_of_blocks_edges ((number_of_edges/threads_per_block) + 1)

struct Edge{

    int flow;
    int capacity;

    Edge(int f, int c){

        this->flow = f;
        this->capacity = c;
    }
};

struct Vertex{

    int height;
    int excess_flow;

    Vertex(){

        this->height = 0;
        this->excess_flow = 0;
    }
};

class Graph{

    int V;
    int excess_total;
    Vertex* node_list;
    Edge** adj_mtx;

};

void read_graph(Edge** adj_mtx,int V){

    FILE *file_pointer = fopen("adjacency_matrix.txt","r");
    
    // char buf1[5],buf2[5],buf3[5];

     // read file and fill capacity values of adj_mtx, let flow = 0
     // see graph generator fn and write accordingly
     // CODE MISSING !!!

}

__global__ void preflow_kernel(Graph* gpu_graph, int source){

    unsigned int id = (blockIdx.x*blockDim.x) + threadIdx.x;
    
    if(id < gpu_graph->V){

        gpu_graph->node_list[source].height = gpu_graph->V;

        if(gpu_graph->adj_mtx[source][id].capacity > 0){

            gpu_graph->adj_mtx[source][id].flow = gpu_graph->adj_mtx[source][id].capacity;
            atomicAdd(&(gpu_graph->node_list[id].excess_flow), gpu_graph->adj_mtx[source][id].flow);
            atomicSub(&(gpu_graph->adj_mtx[id][source].flow), gpu_graph->adj_mtx[source][id].flow); // !!! Not sure if right
            // What about excess total???
            // What about source's excess flow???

        }

    }    

}


int main(int argc, char** argv){

    if(argc < 5){
        printf("Insufficient number of arguments in CLI\n");
        exit(0);
    }

    int V = atoi(argv[1]);
    int E = atoi(argv[2]);
    int source = atoi(argv[3]);
    int sink = atoi(argv[4]);

    Graph *cpu_graph,*gpu_graph;
    Vertex *cpu_node_list,*gpu_node_list;
    Edge **cpu_adj_mtx,**gpu_adj_mtx;

    cpu_graph = (Graph*)malloc(sizeof(Graph));
    cpu_node_list = (Vertex*)malloc(V*sizeof(Vertex));
    cpu_adj_mtx = (Edge**)malloc(V*V*sizeof(Edge));

    cpu_graph->V = V;
    cpu_graph->excess_total = 0;
    cpu_graph->node_list = cpu_node_list;
    cpu_graph->adj_mtx = cpu_adj_mtx;

    cudaMalloc((void**)&gpu_graph,sizeof(graph));
    cudaMalloc((void**)&gpu_node_list,V*sizeof(Vertex));
    cudaMalloc((void***)&gpu_adj_mtx,V*V*sizeof(Edge));


    // read_graph()

    cudaMemcpy(gpu_graph,cpu_graph,sizeof(Graph),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_node_list,cpu_node_list,V*sizeof(Vertex),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_adj_mtx,cpu_adj_mtx,V*V*sizeof(Edge),cudaMemcpyHostToDevice);
    
    cudaMemcpy(&(gpu_graph->node_list),&gpu_node_list,sizeof(Vertex*),cudaMemcpyHostToDevice);
    cudaMemcpy(&(gpu_graph->adj_mtx),&gpu_adj_mtx,sizeof(Edge**),cudaMemcpyHostToDevice);

    // time start 

    // preflow invoke - from Host or Device ??

    // push relabel host fn - which has while routine - which has the push relabel kernel - and global relabel host fn

    // time end

    cudaMemcpy(cpu_graph,gpu_graph,sizeof(Graph),cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_node_list,gpu_node_list,V*sizeof(Vertex),cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_adj_mtx,gpu_adj_mtx,V*V*sizeof(Edge),cudaMemcpyDeviceToHost);

    cpu_graph->node_list = cpu_node_list;
    cpu_graph->adj_mtx = cpu_adj_mtx;

    // print result

    return 0;
}

