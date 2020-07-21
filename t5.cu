#include<cuda.h>
#include<bits/stdc++.h>

#define number_of_nodes V
#define number_of_edges E
#define threads_per_block 256
#define number_of_blocks_nodes ((number_of_nodes/threads_per_block) + 1)
#define number_of_blocks_edges ((number_of_edges/threads_per_block) + 1)
#define INF 1000000000
#define IDX(x,y) ( ( (x)*(number_of_nodes) ) + (y) )
#define KERNEL_CYCLES number_of_blocks_nodes

void readgraph(int V, int E, int source, int sink, int *cpu_height, int *cpu_excess_flow, int *cpu_adjmtx, int *cpu_rflowmtx)
{
    // initialising all adjacent matrix values to 0 before input 
    for(int i = 0; i < (number_of_nodes)*(number_of_nodes); i++)
    {
        cpu_adjmtx[i] = 0;
        cpu_rflowmtx[i] = 0;
    }
    // declaring file pointer to read edgelist
    FILE *fp = fopen("edgelist.txt","r");

    // declaring variables to read and store data from file
    char buf1[10],buf2[10],buf3[10];
    int e1,e2,cp;

    // getting edgelist input from file "edgelist.txt"
    for(int i = 0; i < E; i++)
    {
        // reading from file
        fscanf(fp,"%s",buf1);
        fscanf(fp,"%s",buf2);
        fscanf(fp,"%s",buf3);

        // storing as integers
        e1 = atoi(buf1);
        e2 = atoi(buf2);
        cp = atoi(buf3);

        /* Adding edge to graph if it does not have source as to node, or sink as from node
         * rflow - residual flow is also updated simultaneously
         * So the graph when prepared already has updated residual flow values
         * This is why residual flow is not initialised during preflow
         */

        if( (e2 != source) && (e1 != sink) )
        {
            cpu_adjmtx[IDX(e1,e2)] = cp;
            cpu_rflowmtx[IDX(e1,e2)] = cp;    
        }

    }

}

void preflow(int V, int source, int sink, int *cpu_height, int *cpu_excess_flow, int *cpu_adjmtx, int *cpu_rflowmtx, int *Excess_total)
{
    // initialising height values and excess flow, Excess_total values
    for(int i = 0; i < V; i++)
    {
        cpu_height[i] = 0; 
        cpu_excess_flow[i] = 0;
    }
    
    cpu_height[source] = V;
    *Excess_total = 0;

    // pushing flow in all edges going out from the source node
    for(int i = 0; i < V; i++)
    {
        // for all (source,i) belonging to E :
        if(cpu_adjmtx[IDX(source,i)] > 0)
        {
            // pushing out of source node
            cpu_rflowmtx[IDX(source,i)] = 0;
            
            /* updating the residual flow value on the back edge
             * u_f(x,s) = u_xs + u_sx
             * The capacity of the back edge is also added to avoid any push operation back to the source 
             * This avoids creating a race condition, where flow keeps travelling to and from the source
             */
            cpu_rflowmtx[IDX(i,source)] = cpu_adjmtx[IDX(source,i)] + cpu_adjmtx[IDX(i,source)];
            
            // updating the excess flow value of the node flow is pushed to, from the source
            cpu_excess_flow[i] = cpu_adjmtx[IDX(source,i)];

            // update Excess_total value with the new excess flow value of the node flow is pushed to
            *Excess_total += cpu_excess_flow[i];
        } 
    }

}

__global__ void push_relabel_kernel(int V, int *gpu_height, int *gpu_excess_flow, int *gpu_adjmtx,int *gpu_rflowmtx)
{
    // u'th node is operated on by the u'th thread
    unsigned int u = (blockIdx.x*blockDim.x) + threadIdx.x;

    if(u < V)
    {
        // cycle value is set to KERNEL_CYCLES as required 
        int cycle = KERNEL_CYCLES;

        /* Variables declared to be used inside the kernel :
        * e_dash - initial excess flow of node u
        * h_dash - height of lowest neighbor of node u
        * h_double_dash - used to iterate among height values to find h_dash
        * v - used to iterate among nodes to find v_dash
        * v_dash - lowest neighbor of node u 
        * d - flow to be pushed from node u
        */

        int e_dash,h_dash,h_double_dash,v,v_dash,d;

        while(cycle > 0)
        {
            if( (gpu_excess_flow[u] > 0) && (gpu_height[u] < V) )
            {
                e_dash = gpu_excess_flow[u];
                h_dash = INF;

                for(v = 0; v < V; v++)
                {
                    // for all (u,v) belonging to E_f (residual graph edgelist)
                    if(cpu_rflowmtx[IDX(u,v)] > 0)
                    {
                        h_double_dash = gpu_height[v];
                        // finding lowest neighbor of node u
                        if(h_double_dash < h_dash)
                        {
                            v_dash = v;
                            h_dash = h_double_dash;
                        }
                    }
                }

                if(gpu_height[u] > h_dash)
                {
                    /* height of u > height of lowest neighbor
                    * Push operation can be performed from node u to lowest neighbor
                    * All addition, subtraction and minimum operations are done using Atomics
                    * This is to avoid anomalies in conflicts between multiple threads
                    */

                    // d captures flow to be pushed 
                    d = e_dash;
                    atomicMin(&d,gpu_rflowmtx[IDX(u,v_dash)]);

                    // Residual flow towards lowest neighbor from node u is increased
                    atomicAdd(&cpu_rflowmtx[IDX(v_dash,u)],d);

                    // Residual flow towards node u from lowest neighbor is decreased
                    atomicSub(&cpu_rflowmtx[IDX(u,v_dash)],d);

                    // Excess flow of lowest neighbor and node u are updated
                    atomicAdd(&gpu_excess_flow[v_dash],d);
                    atomicSub(&gpu_excess_flow[u],d);
                }

                else
                {
                    /* height of u <= height of lowest neighbor,
                    * No neighbor with lesser height exists
                    * Push cannot be performed to any neighbor
                    * Hence, relabel operation is performed
                    */

                    gpu_height[u] = h_dash + 1;
                }

            }

            // cycle value is decreased
            cycle = cycle - 1;

        }
    }
}

void push_relabel(int V, int source, int sink, int *cpu_height, int *cpu_excess_flow, int *cpu_adjmtx, int *cpu_rflowmtx, int *Excess_total, int *gpu_height, int *gpu_excess_flow, int *gpu_adjmtx, int *gpu_rflowmtx)
{
    /* Instead of checking for overflowing vertices(as in the sequential push relabel),
     * sum of excess flow values of sink and source are compared against Excess_total 
     * If the sum is lesser than Excess_total, 
     * it means that there is atleast one more vertex with excess flow > 0, apart from source and sink
     */

    while((cpu_excess_flow[source] + cpu_excess_flow[sink]) < *Excess_total)
    {
        // copying height values to CUDA device global memory
        cudaMemcpy(gpu_height,cpu_height,V*sizeof(int),cudaMemcpyHostToDevice);

        // invoking the push_relabel_kernel
        push_relabel_kernel<<<number_of_blocks_nodes,threads_per_block>>>(V,gpu_height,gpu_excess_flow,gpu_adjmtx,gpu_rflowmtx);

        // copying height, excess flow and residual flow values from device to host memory
        cudaMemcpy(cpu_height,gpu_height,V*sizeof(int),cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_excess_flow,gpu_excess_flow,V*sizeof(int),cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_rflowmtx,gpu_rflowmtx,V*V*sizeof(int),cudaMemcpyDeviceToHost);

        // perform the global_relabel routine on host
        global_relabel();

    }

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

    // time start

    // invoking the preflow function to initialise values in host
    preflow(V,source,sink,cpu_height,cpu_excess_flow,cpu_adjmtx,cpu_rflowmtx,Excess_total);

    // copying host data to CUDA device global memory
    cudaMemcpy(gpu_height,cpu_height,V*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_excess_flow,cpu_excess_flow,V*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_adjmtx,cpu_adjmtx,V*V*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_rflowmtx,cpu_rflowmtx,V*V*sizeof(int),cudaMemcpyHostToDevice);

    // push_relabel()
    push_relabel(V,source,sink,cpu_height,cpu_excess_flow,cpu_adjmtx,cpu_rflowmtx,Excess_total,gpu_height,gpu_excess_flow,gpu_adjmtx,gpu_rflowmtx)
    
    // print value
    printf("The maximum flow value of this flow network is %d\n",cpu_excess_flow[sink]);

    // free host memory

    // free device memory


    return 0;

}
