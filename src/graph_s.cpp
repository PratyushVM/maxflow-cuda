#include "../include/serial_graph.h"

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

// check function to return value from serial implementation
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


