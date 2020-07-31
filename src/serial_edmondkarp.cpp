/*
 * Implementation of serial Ford-Fulkerson (Edmond-Karp's) algorithm in C++
 * Used to verify correctness of output of parallel implementation of max-flow problem in CUDA
 * 
 */

#include<bits/stdc++.h>
using namespace std;

#define number_of_nodes 6

/*
 * Function : bfs
 * Arguments : adjacency matrix of residual graph, number of nodes, source, sink, parent array
 * Returns whether there exists a path between the source and the sink
 * Updates the parent array which can be used to retrace the path
 * 
 */

bool bfs(int residual_graph[number_of_nodes][number_of_nodes], int source, int sink, int parent[number_of_nodes])
{
    // Create a boolean array to track nodes visited and initialize all nodes to 0
    bool visited[number_of_nodes];
    memset(visited,0,sizeof(visited));

    // Create a queue for the bfs routine and enqueue the source vertex
    queue<int> bfs_queue;
    bfs_queue.push(source);
    visited[source] = true;
    parent[source] = -1;

    // bfs routine that also updates parent array to trace the augmented path
    while(!bfs_queue.empty())
    {
        // dequeue vertex at front
        int u = bfs_queue.front();
        bfs_queue.pop();

        // enqueue all unvisited vertices connected to the dequeued vertex
        for(int v = 0; v < number_of_nodes; v++)
        {
            if(visited[v] == false && residual_graph[u][v] > 0)
            {
                bfs_queue.push(v);
                visited[v] = true;
                parent[v] = u;
            }
        }

    }

    // If the sink is reached from the source, then augmented path from source to sink exists
    return (visited[sink] == true);
}


/* 
 * Function : maxflow
 * Arguments : adjacency matrix of given graph(flow network), number of nodes, source, sink
 * Returns the maximum network flow of the given flow network
 * 
 */

int maxflow(int graph[number_of_nodes][number_of_nodes], int source, int sink)
{
    int u,v;

    // Create a residual graph and fill it with given capacities of original flow network
    int residual_graph[number_of_nodes][number_of_nodes];
    for(u = 0; u < number_of_nodes; u++)
    {
        for(v = 0; v < number_of_nodes; v++)
        {
            residual_graph[u][v] = graph[u][v];
        }
    }

    // declare parent array and max_flow of the flow network
    int parent[number_of_nodes];
    int max_flow = 0;

    // Augment flow while there exists a path from source to sink
    while(bfs(residual_graph,source,sink,parent))
    {
        // Find minimum residual capacity along the augmented path - this is the maximum flow across the augmented path
        int path_flow = INT_MAX;

        for(v = sink; v != source; v = parent[v])
        {
            u = parent[v];
            path_flow = min(path_flow,residual_graph[u][v]);
        }

        // Update the residual capacities across the augmented path, and reverse edges accordingly
        for(v = sink; v != source; v = parent[v])
        {
            u = parent[v];
            residual_graph[u][v] -= path_flow;
            residual_graph[v][u] += path_flow;
        }

        // Add the flow of the augmented path to the max_flow of the network
        max_flow += path_flow;
    }

    // Return total max_flow of the given flow network
    return max_flow;
}

// main/check function

int main()
{
    int graph[number_of_nodes][number_of_nodes] = { {0, 16, 13, 0, 0, 0}, 
                        {0, 0, 10, 12, 0, 0}, 
                        {0, 4, 0, 0, 14, 0}, 
                        {0, 0, 9, 0, 0, 20}, 
                        {0, 0, 0, 7, 0, 4}, 
                        {0, 0, 0, 0, 0, 0} 
                      }; 
  

    cout<<"The maximum flow of the network is "<<maxflow(graph,0,5); 

    return 0;
}

