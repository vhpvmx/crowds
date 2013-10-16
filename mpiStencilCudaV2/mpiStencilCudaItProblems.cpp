#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <mpi.h>
#include <math.h>
#include <stddef.h>
//Compilation: make
//Local Execution: optirun mpirun -np 4 ./mpiStencilCuda

//HPV  
#define DATA_COLLECT 3
#define RADIO 1
#define DEBUG

#define agent_width 3
#define agent_height 3
#define world_width 96
#define world_height 96
#define nreps 3
int agents_total =  agent_width * agent_height;
//#define agents_total = agent_width * agent_height

struct sAgents{
	float4 pos[agent_width * agent_height];
	float4 ids[agent_width * agent_height];
};


extern "C" void launch_kernel(float4 *d_agents_pos, float4 *d_agents_ids, int *d_world, int world_width_a, int world_height_a, int agent_width_a, int agent_height_a, int world_height_node, int pid);  
	
float Ranf( float, float );

// Initialize agents with random positions
void init_data(sAgents &h_agents_in)
{
	const float PI = 3.1415926535;
	const float XMIN = 	{ 2.0 };
	const float XMAX = 	{  94.0 };
	const float YMIN = 	{ 2.0 };
	const float YMAX = 	{  94.0 };
	const float THMIN =	{   0.0};
	const float THMAX =	{    2*PI }; //2*PI
	const float VMIN =	{   1.0 }; //0.0
	const float VMAX =	{    2.0 };

	for(int i = 0; i< agents_total;i++)
	{
		h_agents_in.ids[i].x = i + 1;
		h_agents_in.ids[i].y = 0;
		h_agents_in.ids[i].z = 0;
		h_agents_in.pos[i].x = round(Ranf( XMIN, XMAX ));
		h_agents_in.pos[i].y = round(Ranf( YMIN, YMAX ));
		h_agents_in.pos[i].z = Ranf( THMIN, THMAX );
		h_agents_in.pos[i].w = Ranf( VMIN, VMAX );
	}	
}

void display_data(sAgents h_agents_in)
{
    for( int i = 0; i < agents_total; i++ )
	printf("id: %f pid: %f status: %f x: %f y: %f z: %f, w: %f \n", h_agents_in.ids[i].x, h_agents_in.ids[i].z, h_agents_in.ids[i].y, h_agents_in.pos[i].x, h_agents_in.pos[i].y, h_agents_in.pos[i].z, h_agents_in.pos[i].w);
}

void data_server()
{
	int np; 
	MPI_Comm_size(MPI_COMM_WORLD, &np);
	MPI_Status status;

	int num_comp_nodes = np -1;
	unsigned int num_bytes = sizeof(sAgents);
	sAgents h_agents_in, h_agents_out[num_comp_nodes];

	/* initialize input data */
	init_data(h_agents_in);

	#ifdef DEBUG 
	printf("Init data\n");
	display_data(h_agents_in);
	#endif

	/* send data to compute nodes */
	for(int process = 0; process < num_comp_nodes; process++)
		MPI_Send(&h_agents_in, num_bytes, MPI_BYTE, process, 0, MPI_COMM_WORLD);
		
	for(int it = 0; it < nreps ; it++)
	{
		/* Wait for nodes to compute */
		MPI_Barrier(MPI_COMM_WORLD);

		for(int process = 0; process < num_comp_nodes; process++)
		{

			MPI_Recv(&h_agents_out[process], num_bytes, MPI_BYTE, process, DATA_COLLECT, MPI_COMM_WORLD, &status); 

			for( int i = 0; i < agents_total; i++)
			{
				if( h_agents_out[process].ids[i].y == 1 )
				{
					h_agents_in.pos[i] = h_agents_out[process].pos[i]; 
					h_agents_in.ids[i] = h_agents_out[process].ids[i];
				}
			}
	
		}
		#ifdef DEBUG
	        printf("Final Data it:%d\n", it );	
		display_data(h_agents_in);
		#endif

	}



	
	/* release resources */
	//free(&h_agents_in);
	//free(&h_agents_out); 
}

void compute_process()
{
	int np, pid;
	MPI_Comm_rank(MPI_COMM_WORLD, &pid);
	MPI_Comm_size(MPI_COMM_WORLD, &np);
	int server_process = np - 1;
	MPI_Status status;

	int num_comp_nodes = np -1;
	unsigned int num_bytes = sizeof(sAgents);
	unsigned int num_halo_points = RADIO * world_width;
	unsigned int num_halo_bytes = num_halo_points * sizeof(int);
 
	size_t size_world = world_width * world_height * sizeof(int);
	int *h_world = (int *)malloc(size_world);
	int *d_world;

	int left_neighbor = (pid > 0) ? (pid - 1) : MPI_PROC_NULL;
	int right_neighbor = (pid < np -2) ? (pid + 1) : MPI_PROC_NULL;


	for(int j = 0; j < world_width * world_height; j++)
	{	
		h_world[j] = 0;
	}


	sAgents h_agents_in, h_agents_left_node, h_agents_right_node;
	float4 h_agents_pos[agents_total], h_agents_ids[agents_total];
	float4 *d_agents_pos, *d_agents_ids;
	unsigned int num_bytes_agents = agents_total * sizeof(float4);

	int world_height_node = world_height / num_comp_nodes;

	// Error code to check return values for CUDA calls
        cudaError_t err = cudaSuccess;

	// Allocate the device pointer
    	err = cudaMalloc((void **)&d_world, size_world);

	if (err != cudaSuccess)
	{
        	fprintf(stderr, "Failed to allocate device pointer (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }

    	err = cudaMalloc((void **)&d_agents_pos, num_bytes_agents);

	if (err != cudaSuccess)
	{
        	fprintf(stderr, "Failed to allocate device pointer (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }

    	err = cudaMalloc((void **)&d_agents_ids, num_bytes_agents);

	if (err != cudaSuccess)
	{
        	fprintf(stderr, "Failed to allocate device pointer (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
		
	MPI_Recv(&h_agents_in, num_bytes, MPI_BYTE, server_process, 0, MPI_COMM_WORLD, &status);

	for(int i = 0; i < agents_total; i++)
	{
		//identify the active agents according to the y coordinate and set the busy cells in the world
		if(  ( round(h_agents_in.pos[i].y) >= (pid * world_height_node) ) and ( round(h_agents_in.pos[i].y) < ( (pid + 1) * world_height_node ) )  )
		{	
			h_agents_in.ids[i].y = 1;
			h_agents_in.ids[i].z = pid;
			h_world[(int)round( (world_width * (h_agents_in.pos[i].y - 1) ) + h_agents_in.pos[i].x )] = h_agents_in.ids[i].x;
		}
		//Copy the data to a local arrays
		h_agents_pos[i] = h_agents_in.pos[i];
		h_agents_ids[i] = h_agents_in.ids[i];
	}

		#ifdef DEBUG
		printf("antes del ciclo: %d\n", pid);
		display_data(h_agents_in);
		#endif


	err = cudaMemcpy(d_world, h_world, size_world, cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
    	{
        	fprintf(stderr, "Failed to copy pointer from host to device (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}


	for(int it = 0; it < nreps ; it++)
	{
		err = cudaMemcpy(d_agents_pos, h_agents_pos, num_bytes_agents, cudaMemcpyHostToDevice);

		if (err != cudaSuccess)
	    	{
	        	fprintf(stderr, "Failed to copy pointer from host to device (error code %s)!\n", cudaGetErrorString(err));
	        	exit(EXIT_FAILURE);
	    	}
	
		err = cudaMemcpy(d_agents_ids, h_agents_ids, num_bytes_agents, cudaMemcpyHostToDevice);

		if (err != cudaSuccess)
	    	{
	        	fprintf(stderr, "Failed to copy pointer from host to device (error code %s)!\n", cudaGetErrorString(err));
	        	exit(EXIT_FAILURE);
	    	}

		#ifdef DEBUG
		printf("pid: %d\n", pid);
		display_data(h_agents_in);
		#endif

		launch_kernel(d_agents_pos, d_agents_ids, d_world, world_width, world_height, agent_width, agent_height, world_height_node, pid );

		cudaMemcpy(h_agents_pos, d_agents_pos, num_bytes_agents, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_agents_ids, d_agents_ids, num_bytes_agents, cudaMemcpyDeviceToHost);

		printf("%f\n", h_agents_pos[0].y);
		//copy the data to the struct
		for( int i = 0; i < agents_total; i++)
		{
			h_agents_in.pos[i] = h_agents_pos[i];
			h_agents_in.ids[i] = h_agents_ids[i];
		}

		#ifdef DEBUG
		printf("pid: %d\n", pid);
		display_data(h_agents_in);
		#endif


		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Send(&h_agents_in, num_bytes, MPI_BYTE, server_process, DATA_COLLECT, MPI_COMM_WORLD);


		// send data to left, get data from right 
		MPI_Sendrecv(&h_agents_in, num_bytes, MPI_BYTE, left_neighbor, it, &h_agents_right_node, num_bytes, MPI_BYTE, right_neighbor, it, MPI_COMM_WORLD, &status);

		// send data to right, get data from left 
		MPI_Sendrecv(&h_agents_in, num_bytes, MPI_BYTE, right_neighbor, it, &h_agents_left_node, num_bytes, MPI_BYTE, left_neighbor, it, MPI_COMM_WORLD, &status);

		for( int i = 0; i < agents_total; i++)
		{
			if(pid != np-2)
			{
				if(h_agents_right_node.ids[i].y == 2)
				{
					h_agents_in.pos[i] = h_agents_right_node.pos[i];
					h_agents_pos[i] = h_agents_right_node.pos[i];
					h_agents_in.ids[i].y = 1;			
					h_agents_ids[i].y = 1;	
					h_agents_ids[i].z = pid;
				}
			}
			if(pid != 0)				
			{
				if(h_agents_left_node.ids[i].y == 3)
				{
					h_agents_in.pos[i] = h_agents_left_node.pos[i];
					h_agents_pos[i] = h_agents_left_node.pos[i];
					h_agents_in.ids[i].y = 1;			
					h_agents_ids[i].y = 1;	
					h_agents_ids[i].z = pid;
				}
			}
		}

/***
		if(pid == 1)
		{	
			printf("pid: %d\n", pid);
			display_data(h_agents_in);
			display_data(h_agents_right_node);
			display_data(h_agents_left_node);
		}
***/

	}


	/* Release resources */
//	free(h_agents_in); 
/*	
	free(h_output);
	cudaFreeHost(h_left_boundary); cudaFreeHost(h_right_boundary);
	cudaFreeHost(h_left_halo); cudaFreeHost(h_right_halo);
	cudaFree(d_input); cudaFree(d_output);
*/
}


int main(int argc, char *argv[])
{
	int pid = -1, np = -1;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &pid);
	MPI_Comm_size(MPI_COMM_WORLD, &np);


	if(np<3)
	{
		if(0 == pid) printf("Needed 3 or more processes.\n");
		MPI_Abort(MPI_COMM_WORLD, 1); 
		return 1;
	}
	if(pid < np - 1)
		compute_process();
	else
		data_server();

/*
	if(pid < np - 1)
		iterate_process();
	else
		iterate_server();
*/

	MPI_Finalize();
	return 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Random number generator
////////////////////////////////////////////////////////////////////////////////
#define TOP	2147483647.		// 2^31 - 1	
float Ranf( float low, float high )
{
	float r = (float)rand( );
	return(   low  +  r * ( high - low ) / (float)RAND_MAX   );
}

