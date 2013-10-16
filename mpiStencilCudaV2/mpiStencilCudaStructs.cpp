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

typedef struct agent_s
{
	float4 pos;
	float4 id;	
	int id=0;
	int x=0;
	int y=0;
	float z=0;
	float w=0;
}agent;

extern "C" void launch_kernel(float4 *d_agents_pos, short int * d_world, int world_width, int world_height);  
	
float Ranf( float, float );

// Initialize agents with random positions
void init_data(agent *h_agents_in, int agents_total)
{
	const float PI = 3.1415926535;
	const float XMIN = 	{ 1.0 };
	const float XMAX = 	{  96.0 };
	const float YMIN = 	{ 1.0 };
	const float YMAX = 	{  96.0 };
	const float THMIN =	{   0.0};
	const float THMAX =	{    2*PI }; //2*PI
	const float VMIN =	{   0.0 }; //0.0
	const float VMAX =	{    1.0 };

    for( int i = 0; i < agents_total; i++ )
    {
	h_agents_in[ i ].id = i;
	h_agents_in[ i ].x = round(Ranf( XMIN, XMAX ));
	h_agents_in[ i ].y = round(Ranf( YMIN, YMAX ));
	h_agents_in[ i ].z = Ranf( THMIN, THMAX);
	h_agents_in[ i ].w = Ranf( VMIN, VMAX);

	//printf("i: %d x: %f y: %f z: %f, w: %f \n", i, h_dptr[i].x, h_dptr[i].y, h_dptr[i].z, h_dptr[i].w);
    }	
}

void display_data(agent *h_agents_in, int agents_total)
{
    for( int i = 0; i < agents_total; i++ )
	printf("id: %d x: %d y: %d z: %f, w: %f \n", h_agents_in[i].id, h_agents_in[i].x, h_agents_in[i].y, h_agents_in[i].z, h_agents_in[i].w);
}

void data_server(int agents_total, int world_width, int world_height)
{
	int np; 
	MPI_Comm_size(MPI_COMM_WORLD, &np);

	/* create a type for struct agent */
	const int nitems=5;
   	int blocklengths[5] = {1,1,1,1,1};
   	MPI_Datatype types[5] = {MPI_INT, MPI_INT, MPI_INT, MPI_FLOAT, MPI_FLOAT};
	MPI_Datatype mpi_agent_type;
	MPI_Aint offsets[5];

	offsets[0] = offsetof(agent, id);
    	offsets[1] = offsetof(agent, x);
    	offsets[2] = offsetof(agent, y);
    	offsets[3] = offsetof(agent, z);
    	offsets[4] = offsetof(agent, w);

	MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_agent_type);
	MPI_Type_commit(&mpi_agent_type);


	int num_comp_nodes = np -1;
	unsigned int num_bytes = agents_total * sizeof(agent);
	agent *h_agents_in, *h_agents_out;

	/* allocate input data */
	h_agents_in = (agent *)malloc(num_bytes);
	h_agents_out = (agent *)malloc(num_bytes);
	if(h_agents_in == NULL || h_agents_out == NULL)
	{
		printf("server couldn't allocate memory\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	/* initialize input data */
	init_data(h_agents_in, agents_total);

#ifdef DEBUG 
	printf("Init data\n");
	display_data(h_agents_in, agents_total);
#endif

	int world_height_node = world_height / num_comp_nodes;
//	printf("world_height: %d\n", world_height_node);
	agent h_agents_node_in[num_comp_nodes][agents_total], h_agents_node_out[num_comp_nodes][agents_total];
	for(int process = 0; process < num_comp_nodes; process++)
	{	
		for(int i = 0; i < agents_total; i++)
		{
			if(  ( h_agents_in[i].y >= (process * world_height_node) ) and ( h_agents_in[i].y < ( (process + 1) * world_height_node ) )  )
				h_agents_node_in[process][i] = h_agents_in[i];
		}
	}

/***	
	printf("copy data 0\n");
	display_data(h_agents_node_in[0], agents_total);
	printf("copy data 1\n");
	display_data(h_agents_node_in[1], agents_total);
	printf("copy data 2\n");
	display_data(h_agents_node_in[2], agents_total);
***/

	/* send data to compute nodes */
	for(int process = 0; process < num_comp_nodes; process++)
		MPI_Send(h_agents_node_in[process], agents_total, mpi_agent_type, process, 0, MPI_COMM_WORLD);

	/* Wait for nodes to compute */
	MPI_Barrier(MPI_COMM_WORLD);
	
	/* Collect output data */
	MPI_Status status;

	for(int process = 0; process < num_comp_nodes; process++)
		MPI_Recv(h_agents_node_out[process], agents_total, mpi_agent_type, process, DATA_COLLECT, MPI_COMM_WORLD, &status); 

#ifdef DEBUG
        printf("Final Data\n");	
	/* display output data */
//	display_data(h_agents_out, agents_total);
#endif
	
	/* release resources */
	free(h_agents_in);
	free(h_agents_out); 
//	free(h_agents_node_in); 
//	free(h_agents_node_out); 
}

void compute_process(int agents_total, int nreps, int world_width, int world_height)
{
	int np, pid;
	MPI_Comm_rank(MPI_COMM_WORLD, &pid);
	MPI_Comm_size(MPI_COMM_WORLD, &np);
	int server_process = np - 1;
	MPI_Status status;

	/* create a type for struct agent */
	const int nitems=5;
   	int blocklengths[5] = {1,1,1,1,1};
   	MPI_Datatype types[5] = {MPI_INT, MPI_INT, MPI_INT, MPI_FLOAT, MPI_FLOAT};
	MPI_Datatype mpi_agent_type;
	MPI_Aint offsets[5];

	offsets[0] = offsetof(agent, id);
    	offsets[1] = offsetof(agent, x);
    	offsets[2] = offsetof(agent, y);
    	offsets[3] = offsetof(agent, z);
    	offsets[4] = offsetof(agent, w);

	MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_agent_type);
	MPI_Type_commit(&mpi_agent_type);

	unsigned int num_bytes = agents_total * sizeof(float4);
	unsigned int num_halo_points = RADIO * world_width;
	unsigned int num_halo_bytes = num_halo_points * sizeof(short int);

	//unsigned int world_node_height = (world_height / (np-1)) + (RADIO * 2);
	//if(pid == 0 or pid == np - 2)
	//	world_node_height -= RADIO;
 
	size_t size_world = world_width * world_height * sizeof(short int);
	short int *h_world = (short int *)malloc(size_world);
	*h_world = 0;
	short int *d_world;

	for(int j = 0; j < world_width * world_height; j++)
	{	
		h_world[j] = 0;
	}

	/* alloc host memory */
	agent *h_agents_in = (agent *)malloc(num_bytes);
	//agent *d_agents_in;
	float4 *h_agents_pos;
	float4 *d_agents_pos;
	
	
	//MPI_Recv(rcv_address, num_points, MPI_FLOAT, server_process, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
	MPI_Recv(h_agents_in, agents_total, mpi_agent_type, server_process, 0, MPI_COMM_WORLD, &status);

	//Iniatialize world
	for( int i = 0; i < agents_total; i++)
	{
		h_world[(world_width * (h_agents_in[i].y - 1) ) + h_agents_in[i].x] = (h_agents_in[i].x!=0?1:0);
		//if(h_world[(world_width * (h_agents_in[i].y - 1) ) + h_agents_in[i].x] == 1)
			//printf("world x: %d, y: %d\n", h_agents_in[i].x, h_agents_in[i].y);	
		h_agents_pos[i].x = h_agents_in[i].x;
		h_agents_pos[i].y = h_agents_in[i].y;
		h_agents_pos[i].z = h_agents_in[i].z;
		h_agents_pos[i].w = h_agents_in[i].w;
	}

/***
	if(pid ==1)
{
	int k=0;
	for(int j = 0; j < world_width * world_height; j++)
	{	
		if ( j%96 == 0 and j>0)
		{
			k++;
			printf("%d row: %d\n", h_world[j], k);
		}
		else
			printf("%d ", h_world[j]);
	}
}
***/

	// Error code to check return values for CUDA calls
        cudaError_t err = cudaSuccess;

	// Allocate the device pointer
    	err = cudaMalloc((void **)&d_world, size_world);

	if (err != cudaSuccess)
	{
        	fprintf(stderr, "Failed to allocate device pointer (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }

	err = cudaMemcpy(d_world, h_world, size_world, cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
    	{
        	fprintf(stderr, "Failed to copy pointer from host to device (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}


	//http://cuda-programming.blogspot.com.es/2013/02/cuda-array-in-cuda-how-to-use-cuda.html
	//http://stackoverflow.com/questions/17924705/structure-of-arrays-vs-array-of-structures-in-cuda
	// Allocate the device pointer

    	err = cudaMalloc((void **)&d_agents_pos, num_bytes);

	if (err != cudaSuccess)
	{
        	fprintf(stderr, "Failed to allocate device pointer (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }

	err = cudaMemcpy(d_agents_pos, h_agents_pos, num_bytes, cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
    	{
        	fprintf(stderr, "Failed to copy pointer from host to device (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}


	launch_kernel(d_agents_pos, d_world, world_width, world_height );

	MPI_Barrier( MPI_COMM_WORLD);

#ifdef DEBUG
//	printf("pid: %d\n", pid);
//	display_data(h_agents_in, agents_total );
#endif

	MPI_Send(h_agents_in, agents_total, mpi_agent_type, server_process, DATA_COLLECT, MPI_COMM_WORLD);


	/* Release resources */
	free(h_agents_in); 
/*	
	free(h_output);
	cudaFreeHost(h_left_boundary); cudaFreeHost(h_right_boundary);
	cudaFreeHost(h_left_halo); cudaFreeHost(h_right_halo);
	cudaFree(d_input); cudaFree(d_output);
*/
}


int main(int argc, char *argv[])
{
	const unsigned int world_width  = 96;
	const unsigned int world_height = 96;
	const unsigned int agents_total = 9;
	int nreps = 3;
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
		compute_process( agents_total, nreps, world_width, world_height );
	else
		data_server( agents_total, world_width, world_height );

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

