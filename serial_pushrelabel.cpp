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
struct Edge
{
    int u,v; // edge from node u to node v
    int flow; // current flow
    int capacity; // capacity of edge

    // constructor function 
    Edge(int f, int c, int a, int b)
    {
        this->u = a;
        this->v = b;
        this->capacity = c;
        this->flow = f;
    }
};

class Graph
{
    int V; // number of vertices
    vector<Vertex> vertex; // vector of vertices
    vector<Edge> edge; // vector of edges

    // function to push excess flow from u
    bool push(int u);

    // function to relabel a vertex u
    void relabel(int u);

    // function to initialize preflow
    void preflow(int s);

    // function to reverse edge
    void updatereverseflow(int i, int flow);

public:
    Graph(int v); // constructor to create graph with v vertices

    void addedge(int u, int v, int w); // function to add an edge

    int maxflow(int s, int t); // function that returns maximum flow from source s to sink t

};

Graph::Graph(int v)
{
    this->V = v;

    // all vertices are initialized with zero height and excess flow
    for(int i = 0; i < V; i++)
    {
        vertex.pb(Vertex(0,0));
    }
}

void Graph::addedge(int u, int v, int capacity)
{
    // flow is initially 0 for all edges
    edge.pb(Edge(0,capacity,u,v));
}

void Graph::preflow(int s)
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
            edge.pb(Edge(-edge[i].flow,0,edge[i].v,s));
        }
    }
}

// function that returns index of overflowing Vertex
int overflowvertex(vector<Vertex>& ver)
{
    for(int i = 1; i < ver.size() - 1; i++)
    {
        if(ver[i].ex_flow > 0)
        return i;
    }

    // return -1 if no overflowing vertex exists
    return -1;
}

// Update reverse flow for flow added on i-th edge
void Graph::updatereverseflow(int i, int flow)
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
    edge.pb(Edge(0,flow,u,v));
}

// To push flow from overflowing vertex u
bool Graph::push(int u)
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
void Graph::relabel(int u)
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
int Graph::maxflow(int s, int t)
{
    preflow(s);

    while(overflowvertex(vertex) != -1)
    {
        int u = overflowvertex(vertex);
        if(!push(u))
        {
            relabel(u);
        }
    }

    // vertex.back() returns final vertex, whose ex_flow will be final maximum flow
    return vertex.back().ex_flow;
}

// Driver program to test above functions 
int main() 
{ 
    int V = 6; 
    Graph g(V); 
  
    // Creating above shown flow network 
    g.addedge(0, 1, 16); 
    g.addedge(0, 2, 13); 
    g.addedge(1, 2, 10); 
    g.addedge(2, 1, 4); 
    g.addedge(1, 3, 12); 
    g.addedge(2, 4, 14); 
    g.addedge(3, 2, 9); 
    g.addedge(3, 5, 20); 
    g.addedge(4, 3, 7); 
    g.addedge(4, 5, 4); 
  
    // Initialize source and sink 
    int s = 0, t = 4; 
  
    cout << "Maximum flow is " << g.maxflow(s, t); 
    return 0; 
} 


