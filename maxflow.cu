#include<cuda.h>
#include<bits/stdc++.h>

#define number_of_nodes V
#define number_of_edges E
#define threads_per_block 256
#define number_of_blocks_nodes ((number_of_nodes/threads_per_block) + 1)
#define number_of_blocks_edges ((number_of_edges/threads_per_block) + 1)
#define INF 1000000000
#define IDX(x,y) ( ( (x)*(number_of_nodes) ) + (y) )
#define KERNEL_CYCLES V

#include<bits/stdc++.h>
using namespace std;

#define vi vector<int>
#define pb push_back
#define pii pair<int,int>
#define mp make_pair
#define ff first
#define ss second

// Structure representing a Vertex
struct Vertex
{
    int height; // height of node
    int ex_flow; // excess flow 

    // constructor function
    Vertex(int h, int e)
    {
        this->height = h;
        this->ex_flow = e;
    }
};

// Structure representing an Edge
struct Edge_s
{
    int u,v; // edge from node u to node v
    int flow; // current flow
    int capacity; // capacity of edge

    // constructor function 
    Edge_s(int f, int c, int a, int b)
    {
        this->u = a;
        this->v = b;
        this->capacity = c;
        this->flow = f;
    }
};

class Graph_s
{
    int V; // number of vertices
    vector<Vertex> vertex; // vector of vertices
    vector<Edge_s> edge; // vector of edges

    // function to push excess flow from u
    bool push(int u);

    // function to relabel a vertex u
    void relabel(int u);

    // function to initialize preflow
    void preflow(int s);

    // function to reverse edge
    void updatereverseflow(int i, int flow);

public:
    Graph_s(int v); // constructor to create graph with v vertices

    void addedge(int u, int v, int w); // function to add an edge

    int maxflow(int s, int t); // function that returns maximum flow from source s to sink t

};

Graph_s::Graph_s(int v)
{
    this->V = v;

    // all vertices are initialized with zero height and excess flow
    for(int i = 0; i < V; i++)
    {
        vertex.pb(Vertex(0,0));
    }
}

void Graph_s::addedge(int u, int v, int capacity)
{
    // flow is initially 0 for all edges
    edge.pb(Edge_s(0,capacity,u,v));
}

void Graph_s::preflow(int s)
{
    // making height of source vertex equal to number of vertices
    // height of other vertices are 0 by default
    vertex[s].height = vertex.size();

    for(int i = 0; i < edge.size(); i++)
    {
        // if current edge goes from source
        if(edge[i].u == s)
        {
            // flow is equal to capacity
            edge[i].flow = edge[i].capacity;

            // initialize excess flow for adjacent vertices
            vertex[edge[i].v].ex_flow += edge[i].flow;

            // add reverse edge in residual graph with capacity equal to 0
            edge.pb(Edge_s(-edge[i].flow,0,edge[i].v,s));
        }
    }
}

// function that returns index of overflowing Vertex
int overflowvertex(vector<Vertex>& ver, int s, int t)
{
    for(int i = 0; i < ver.size(); i++)
    {
        if( i != s && i != t && ver[i].ex_flow > 0 )
        return i;
    }

    // return -1 if no overflowing vertex exists
    return -1;
}

// Update reverse flow for flow added on i-th edge
void Graph_s::updatereverseflow(int i, int flow)
{
    int u = edge[i].v , v = edge[i].u;

    for(int j = 0; j < edge.size(); j++)
    {
        if(edge[j].v == v && edge[j].u == u)
        {
            edge[j].flow -= flow;
            return;
        }
    }

    // if reverse edge not present in residual graph
    edge.pb(Edge_s(0,flow,u,v));
}

// To push flow from overflowing vertex u
bool Graph_s::push(int u)
{
    // Traverse through all edges to find an adjacent vertex of u, to which flow can be pushed
    for(int i = 0; i < edge.size(); i++)
    {
        if(edge[i].u == u)
        {
            // if flow is equal to capacity then no push is possible
            if(edge[i].flow == edge[i].capacity)
            continue;

            // checking if height of adjacent vertex is smaller than height of overflowing vertex
            if(vertex[u].height > vertex[edge[i].v].height)
            {
                // flow to be pushed is equal to minimum of remaining flow on edge and excess flow
                int flow = min(edge[i].capacity - edge[i].flow, vertex[u].ex_flow);

                // reduce excess flow for overflowing vertex
                vertex[u].ex_flow -= flow;

                // increase excess flow for adjacent vertex
                vertex[edge[i].v].ex_flow += flow;

                // add residual flow 
                edge[i].flow += flow;

                updatereverseflow(i,flow);

                return true;
            }
        }
    }

    return false;
}

// function to relabel vertex u
void Graph_s::relabel(int u)
{
    // initialize mimimum height of an adjacent
    int mh = INT_MAX;

    // find adjacent with lowest height
    for(int i = 0; i < edge.size(); i++)
    {
        if(edge[i].u == u)
        {
            // if flow is equal to capacity then no relabeling
            if(edge[i].flow == edge[i].capacity)
            continue;

            // update minimum height
            if(vertex[edge[i].v].height < mh)
            {
                mh = vertex[edge[i].v].height;

                // updating height of u
                vertex[u].height = mh + 1; 
            }
        }
    }
}

// function to print the maximum flow from source s to sink t
int Graph_s::maxflow(int s, int t)
{
    preflow(s);

    while(overflowvertex(vertex,s,t) != -1)
    {
        int u = overflowvertex(vertex,s,t);
        if(!push(u))
        {
            relabel(u);
        }
    }

    // ex_flow of the sink will be final maximum flow
    return vertex[t].ex_flow;
}

// Driver program to test above functions 
int check(int V, int E, int source, int sink) 
{ 
    
    Graph_s g(V);   

    FILE *fp = fopen("edgelist.txt","r");

    char buf1[10],buf2[10],buf3[10];
    int e1,e2,cp;

    for(int i = 0; i < E; i++)
    {
        fscanf(fp,"%s",buf1);
        fscanf(fp,"%s",buf2);
        fscanf(fp,"%s",buf3);

        e1 = atoi(buf1);
        e2 = atoi(buf2);
        cp = atoi(buf3);

        g.addedge(e1,e2,cp);
    }
  
    // Initialize source and sink 
    int s = source, t = sink; 
  
    //cout << "Maximum flow is " << g.maxflow(s, t); 
    return g.maxflow(s,t) ; 
} 





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

        //if( (e2 != source) || (e1 != sink) )
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

__global__ void push_relabel_kernel(int V, int source, int sink, int *gpu_height, int *gpu_excess_flow, int *gpu_adjmtx,int *gpu_rflowmtx)
{
    // u'th node is operated on by the u'th thread
    unsigned int u = (blockIdx.x*blockDim.x) + threadIdx.x;

    //printf("u : %d\nV : %d\n",u,V);

    if(u < V)
    {
        printf("Thread id : %d\n",u);
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
            if( (gpu_excess_flow[u] > 0) && (u != sink) )
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


void global_relabel(int V, int source, int sink, int *cpu_height, int *cpu_excess_flow, int *cpu_adjmtx, int *cpu_rflowmtx, int *Excess_total, bool *mark, bool *scanned)
{
    for(int u = 0; u < V; u++)
    {
        for(int v = 0; v < V; v++)
        {
            // for all (u,v) belonging to E
            if(cpu_adjmtx[IDX(u,v)] > 0)
            {
                if(cpu_height[u] > cpu_height[v] + 1)
                {
                    cpu_excess_flow[u] = cpu_excess_flow[u] - cpu_rflowmtx[IDX(u,v)];
                    cpu_excess_flow[v] = cpu_excess_flow[v] + cpu_rflowmtx[IDX(u,v)];
                    cpu_rflowmtx[IDX(v,u)] = cpu_rflowmtx[IDX(v,u)] + cpu_rflowmtx[IDX(u,v)];
                    cpu_rflowmtx[IDX(u,v)] = 0;
                }
            }
        }

        // performing backwards bfs from sink and assigning height values with each vertex's BFS tree level
        
        // declaring the Queue 
        std::list<int> Queue;

        // declaring variables to iterate over nodes for the backwards bfs and to store current tree level
        int x,y,current;
        
        // initialisation of the scanned array with false, before performing backwards bfs
        for(int i = 0; i < V; i++)
        {
            scanned[i] = false;
        }

        // Enqueueing the sink and set scan(sink) to true 
        Queue.push_back(sink);
        scanned[sink] = true;

        // bfs routine and assigning of height values with tree level values
        while(!Queue.empty())
        {
            // dequeue
            x = Queue.front();
            Queue.pop_front();

            // capture value of current level
            current = cpu_height[x];
            
            // increment current value
            current = current + 1;

            for(y = 0; y < V; y++)
            {
                // for all (y,x) belonging to E_f (residual graph)
                if(cpu_rflowmtx[IDX(y,x)] > 0)
                {
                    // if y is not scanned
                    if(scanned[y] == false)
                    {
                        // assign current as height of y node
                        cpu_height[y] = current;

                        // mark scanned(y) as true
                        scanned[y] = true;

                        // Enqueue y
                        Queue.push_back(y);
                    }
                }
            }

        }

        // declaring and initialising boolean variable for checking if all nodes are relabeled
        bool if_all_are_relabeled = true;

        for(int i = 0; i < V; i++)
        {
            if(scanned[i] == false)
            {
                if_all_are_relabeled = false;
                break;
            }
        }

        // if not all nodes are relabeled
        if(if_all_are_relabeled == false)
        {
            // for all nodes
            for(int i = 0; i < V; i++)
            {
                // if i'th node is not marked or relabeled
                if( !( (scanned[i] == true) || (mark[i] == true) ) )
                {
                    // mark i'th node
                    mark[i] = true;

                    /* decrement excess flow of i'th node from Excess_total
                     * This shows that i'th node is not scanned now and needs to be marked, thereby no more contributing to Excess_total
                     */

                    *Excess_total = *Excess_total - cpu_excess_flow[i];
                }
            }
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

    /* declaring the mark and scan boolean arrays used in the global_relabel routine outside the while loop 
     * This is not to lose the mark values if it goes out of scope and gets redeclared in the next iteration 
     */

    bool *mark,*scanned;
    mark = (bool*)malloc(V*sizeof(bool));
    scanned = (bool*)malloc(V*sizeof(bool));

    // initialising mark values to false for all nodes
    for(int i = 0; i < V; i++)
    {
        mark[i] = false;
    }

    while((cpu_excess_flow[source] + cpu_excess_flow[sink]) < *Excess_total)
    {
        // copying height values to CUDA device global memory
        cudaMemcpy(gpu_height,cpu_height,V*sizeof(int),cudaMemcpyHostToDevice);

        printf("Invoking kernel\n");

        // invoking the push_relabel_kernel
        push_relabel_kernel<<<number_of_blocks_nodes,threads_per_block>>>(V,source,sink,gpu_height,gpu_excess_flow,gpu_adjmtx,gpu_rflowmtx);

        cudaDeviceSynchronize();


        // copying height, excess flow and residual flow values from device to host memory
        cudaMemcpy(cpu_height,gpu_height,V*sizeof(int),cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_excess_flow,gpu_excess_flow,V*sizeof(int),cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_rflowmtx,gpu_rflowmtx,V*V*sizeof(int),cudaMemcpyDeviceToHost);

        printf("After invoking\n");
        print(V,cpu_height,cpu_excess_flow,cpu_rflowmtx,cpu_adjmtx);
        printf("Excess total : %d\n",*Excess_total);
        // perform the global_relabel routine on host
        global_relabel(V,source,sink,cpu_height,cpu_excess_flow,cpu_adjmtx,cpu_rflowmtx,Excess_total,mark,scanned);

        printf("\nAfter global relabel\n");
        print(V,cpu_height,cpu_excess_flow,cpu_rflowmtx,cpu_adjmtx);
        printf("Excess total : %d\n",*Excess_total);
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

    print(V,cpu_height,cpu_excess_flow,cpu_rflowmtx,cpu_adjmtx);

    // time start

    // invoking the preflow function to initialise values in host
    preflow(V,source,sink,cpu_height,cpu_excess_flow,cpu_adjmtx,cpu_rflowmtx,Excess_total);

    print(V,cpu_height,cpu_excess_flow,cpu_rflowmtx,cpu_adjmtx);

    // copying host data to CUDA device global memory
    cudaMemcpy(gpu_height,cpu_height,V*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_excess_flow,cpu_excess_flow,V*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_adjmtx,cpu_adjmtx,V*V*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_rflowmtx,cpu_rflowmtx,V*V*sizeof(int),cudaMemcpyHostToDevice);

    // push_relabel()
    push_relabel(V,source,sink,cpu_height,cpu_excess_flow,cpu_adjmtx,cpu_rflowmtx,Excess_total,gpu_height,gpu_excess_flow,gpu_adjmtx,gpu_rflowmtx);
    
    // print value
    int serial_check = check(V,E,source,sink);

    printf("The maximum flow value of this flow network as calculated by the parallel implementation is %d\n",cpu_excess_flow[sink]);
    printf("The maximum flow of this flow network as calculated by the serial implementation is %d\n",serial_check);
    
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
