#include"../include/parallel_graph.cuh"

__global__ void push_relabel_kernel(int V, int source, int sink, int *gpu_height, int *gpu_excess_flow, int *gpu_adjmtx,int *gpu_rflowmtx)
{
    // u'th node is operated on by the u'th thread
    unsigned int u = (blockIdx.x*blockDim.x) + threadIdx.x;

    //printf("u : %d\nV : %d\n",u,V);

    if(u < V)
    {
        //printf("Thread id : %d\n",u);
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
                v_dash = NULL;

                for(v = 0; v < V; v++)
                {
                    // for all (u,v) belonging to E_f (residual graph edgelist)
                    if(gpu_rflowmtx[IDX(u,v)] > 0)
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
                    //atomicMin(&d,gpu_rflowmtx[IDX(u,v_dash)]);
                    if(e_dash > gpu_rflowmtx[IDX(u,v_dash)])
                    {
                        d = gpu_rflowmtx[IDX(u,v_dash)];
                    }
                    // Residual flow towards lowest neighbor from node u is increased
                    atomicAdd(&gpu_rflowmtx[IDX(v_dash,u)],d);

                    // Residual flow towards node u from lowest neighbor is decreased
                    atomicSub(&gpu_rflowmtx[IDX(u,v_dash)],d);

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