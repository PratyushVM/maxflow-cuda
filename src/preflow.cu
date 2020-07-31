#include"../include/parallel_graph.cuh"

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