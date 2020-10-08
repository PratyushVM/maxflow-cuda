#include"../include/parallel_graph.cuh"
#include"../include/serial_graph.h"

int main(int argc, char **argv)
{
    // checking if sufficient number of arguments (4) are passed in CLI
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
    int *cpu_adjmtx,*gpu_adjmtx;
    int *cpu_rflowmtx,*gpu_rflowmtx;
    
    // allocating host memory
    cpu_height = (int*)malloc(V*sizeof(int));
    cpu_excess_flow = (int*)malloc(V*sizeof(int));
    cpu_adjmtx = (int*)malloc(V*V*sizeof(int));
    cpu_rflowmtx = (int*)malloc(V*V*sizeof(int));
    Excess_total = (int*)malloc(sizeof(int));

    // allocating CUDA device global memory
    cudaMalloc((void**)&gpu_height,V*sizeof(int));
    cudaMalloc((void**)&gpu_excess_flow,V*sizeof(int));
    cudaMalloc((void**)&gpu_adjmtx,V*V*sizeof(int));
    cudaMalloc((void**)&gpu_rflowmtx,V*V*sizeof(int));

    // readgraph
    readgraph(V,E,source,sink,cpu_height,cpu_excess_flow,cpu_adjmtx,cpu_rflowmtx);

    //print(V,cpu_height,cpu_excess_flow,cpu_rflowmtx,cpu_adjmtx);

    // time start

    // invoking the preflow function to initialise values in host
    preflow(V,source,sink,cpu_height,cpu_excess_flow,cpu_adjmtx,cpu_rflowmtx,Excess_total);

    //print(V,cpu_height,cpu_excess_flow,cpu_rflowmtx,cpu_adjmtx);

    // copying host data to CUDA device global memory
    cudaMemcpy(gpu_height,cpu_height,V*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_excess_flow,cpu_excess_flow,V*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_adjmtx,cpu_adjmtx,V*V*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_rflowmtx,cpu_rflowmtx,V*V*sizeof(int),cudaMemcpyHostToDevice);

    // push_relabel()
    push_relabel(V,source,sink,cpu_height,cpu_excess_flow,cpu_adjmtx,cpu_rflowmtx,Excess_total,gpu_height,gpu_excess_flow,gpu_adjmtx,gpu_rflowmtx);
    
    // store value from serial implementation
    int serial_check = check(V,E,source,sink);

    // print values from both implementations
    printf("The maximum flow value of this flow network as calculated by the parallel implementation is %d\n",cpu_excess_flow[sink]);
    printf("The maximum flow of this flow network as calculated by the serial implementation is %d\n",serial_check);
    
    // print correctness check result
    if(cpu_excess_flow[sink] == serial_check)
    {
        printf("Passed correctness check\n");
    }
    else
    {
        printf("Failed correctness check\n");
    }

    // free device memory
    cudaFree(gpu_height);
    cudaFree(gpu_excess_flow);
    cudaFree(gpu_adjmtx);
    cudaFree(gpu_rflowmtx);
    
    // free host memory
    free(cpu_height);
    free(cpu_excess_flow);
    free(cpu_adjmtx);
    free(cpu_rflowmtx);
    free(Excess_total);
    
    // return 0 and end program
    return 0;

}
