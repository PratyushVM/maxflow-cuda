#include"../include/parallel_graph.cuh"

void print(int V,int *cpu_height, int *cpu_excess_flow, int *cpu_rflowmtx, int *cpu_adjmtx)
{
    printf("\nHeight :");
    for(int i = 0; i < V; i++)
    {
        printf("%d ",cpu_height[i]);
    }

    printf("\nExcess flow :");
    for(int i = 0; i < V; i++)
    {
        printf("%d ",cpu_excess_flow[i]);
    }

    printf("\nRflow mtx :\n");
    for(int i = 0; i < V; i++)
    {
        for(int j = 0; j < V; j++)
        {
            printf("%d ", cpu_rflowmtx[IDX(i,j)]);
        }
        printf("\n");
    }

    printf("\nAdj mtx :\n");
    for(int i = 0; i < V; i++)
    {
        for(int j = 0; j < V; j++)
        {
            printf("%d ", cpu_adjmtx[IDX(i,j)]);
        }
        printf("\n");
    }
}


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

            cpu_adjmtx[IDX(e1,e2)] = cp;
            cpu_rflowmtx[IDX(e1,e2)] = cp;    
        

    }

}