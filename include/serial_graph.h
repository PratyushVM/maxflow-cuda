#ifndef __SERIAL__GRAPH__HEADER__
#define __SERIAL__GRAPH__HEADER__

#include<bits/stdc++.h>
using namespace std;

// macros declared 

#define vi vector<int>
#define pb push_back
#define pii pair<int,int>
#define mp make_pair
#define ff first
#define ss second

// Classes, structures and function prototypes for serial implementation

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

// Class for graph object of serial implementation
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

int check(int V, int E, int source, int sink);
 
#endif


