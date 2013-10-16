#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <math.h>

#include <mpi.h>

//HPV  
#define DATA_COLLECT 3
#define RADIO 1
#define DEBUG

//Compilation: make
//Local Execution: optirun mpirun -np 4 ./mpiStencilCuda

////////////////////////////////////////////////////////////////////////////////
// constants
const unsigned int window_width  = 512;
const unsigned int window_height = 512;

const unsigned int mesh_width    = 8;
const unsigned int mesh_height   = 8;


//Random variables to define initial position
const unsigned int num_agents	 = mesh_width * mesh_height;
const float PI = 3.1415926535;
const float XMIN = 	{ 128.0 };
const float XMAX = 	{  384.0 };
const float YMIN = 	{ 128.0 };
const float YMAX = 	{  384.0 };
const float THMIN =	{   0.0};
const float THMAX =	{    2*PI }; //2*PI
const float VMIN =	{   0.0 }; //0.0
const float VMAX =	{    1.0 };

float4 *d_world = NULL;

extern "C" void launch_kernel_edges(float * d_output, float * d_input, int dimx, int dimy, int offset, int radio, cudaStream_t stream);
extern "C" void launch_kernel_internal(float * d_output, float * d_input, int dimx, int dimy, int offset_ini, int offset_fin, cudaStream_t stream);  

float Ranf( float, float );

// Initialize agents with random positions
void init_data(float4 *h_agents_in)
{
    //Compute the size of agents and world
    size_t size_agents = num_agents * sizeof(float4);

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    
    // map OpenGL buffer object for writing from CUDA
    float4 *d_agents;

    //Compute agents initial position
    for( int i = 0; i < num_agents; i++ )
    {
	h_agents_in[ i ].x = round(Ranf( XMIN, XMAX ));
	h_agents_in[ i ].y = round(Ranf( YMIN, YMAX ));
	h_agents_in[ i ].z = Ranf( THMIN, THMAX);
	h_agents_in[ i ].w = Ranf( VMIN, VMAX);

	//printf("i: %d x: %f y: %f z: %f, w: %f \n", i, h_dptr[i].x, h_dptr[i].y, h_dptr[i].z, h_dptr[i].w);
    }

    // Copy the host pointer memory to the device memory
    printf("Copy pointer from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_agents, h_agents_in, size_agents, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy pointer from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
}

void display_data(float4 *agents)
{
    for( int i = 0; i < num_agents; i++ )
	printf("i: %d x: %f y: %f z: %f, w: %f \n", i, h_dptr[i].x, h_dptr[i].y, h_dptr[i].z, h_dptr[i].w);
}


void data_server(int dimx, int dimy)
{
	int np; 
	
	MPI_Comm_size(MPI_COMM_WORLD, &np);

	int num_comp_nodes = np -1, first_node = 0, last_node = np - 2;
	unsigned int num_bytes = num_agents * sizeof(float4);
	float4 *h_agents_in, *h_agents_out;

	/* allocate input data */
	h_agents_in = (float4 *)malloc(num_bytes);
	h_agents_out = (float4 *)malloc(num_bytes);
	if(h_agents_in == NULL || h_agents_out == NULL)
	{
		printf("server couldn't allocate memory\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	/* initialize input data */
	init_data(h_agents_in);

#ifdef DEBUG 
//	printf("num_nodos %d\n", np);
	printf("Init data\n");
	/* display input data */ 
	display_data(h_agents_in);
#endif

	/* calculate number of shared points */
	int edge_num_points = dimx * (dimy/num_comp_nodes + RADIO);
	int int_num_points = dimx * (dimy/num_comp_nodes + RADIO * 2);
	float *send_address = input;

/***
#ifdef DEBUG
	int rad = RADIO;
	printf("RADIO %d\n", rad);
	printf("edge_num_points: %d\n", edge_num_points);
	printf("int_num_points: %d\n", int_num_points);	
	printf("send_address_ini: %p\n", send_address);
#endif
***/	

	/* send data to the first compute node */
	MPI_Send(send_address, edge_num_points, MPI_FLOAT, first_node, 0, MPI_COMM_WORLD);

	send_address += dimx * (dimy/num_comp_nodes - RADIO);

/***
#ifdef DEBUG
	printf("send_address_1: %p\n", send_address);
#endif
***/

	/* send data to "internal" compute nodes */
	for(int process = 1; process < last_node; process++)
	{
		MPI_Send(send_address, int_num_points,  MPI_FLOAT, process, 0, MPI_COMM_WORLD);
		send_address += dimx * (dimy/num_comp_nodes);
	}

/***
#ifdef DEBUG
        printf("send_address_fin: %p\n", send_address);
#endif
***/

	/* send data to the last compute node */
	MPI_Send(send_address, edge_num_points, MPI_FLOAT, last_node, 0, MPI_COMM_WORLD);

	/* Wait for nodes to compute */
	MPI_Barrier(MPI_COMM_WORLD);
	
	/* Collect output data */
	MPI_Status status;

	for(int process = 0; process < num_comp_nodes; process++)
		MPI_Recv(output + process * num_points / num_comp_nodes, num_points / num_comp_nodes, MPI_REAL, process, DATA_COLLECT, MPI_COMM_WORLD, &status); 

#ifdef DEBUG
        printf("Final Data\n");	
	/* display output data */
	display_data(output, dimx, dimy);
#endif
	
	/* release resources */
	free(input);
	free(output); 
}

void compute_process(int dimx, int dimy, int nreps)
{
	int np, pid;
	MPI_Comm_rank(MPI_COMM_WORLD, &pid);
	MPI_Comm_size(MPI_COMM_WORLD, &np);
	int server_process = np - 1;
	MPI_Status status;

	unsigned int num_points	= dimx * (dimy + RADIO * 2);
	unsigned int num_bytes = num_points * sizeof(float);
	unsigned int num_halo_points = RADIO * dimx;
	unsigned int num_halo_bytes = num_halo_points * sizeof(float);


	/* alloc host memory */
	float *h_input = (float *)malloc(num_bytes);
	
	/* alloc device memory for input and output data */
	float *d_input = NULL;
	cudaMalloc((void **)&d_input, num_bytes);
	//float *rcv_address = h_input + num_halo_points * (0 == pid);
	float *rcv_address = h_input;
	if(pid == 0)
		rcv_address += num_halo_points;

	//MPI_Recv(rcv_address, num_points, MPI_FLOAT, server_process, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
	MPI_Recv(rcv_address, num_points, MPI_FLOAT, server_process, 0, MPI_COMM_WORLD, &status);
	cudaMemcpy(d_input, h_input, num_bytes, cudaMemcpyHostToDevice);


/***
#ifdef DEBUG
	printf("pid: %d\n", pid);
	display_data(h_input, dimx, (dimy + RADIO * 2) );
#endif
***/

	float *h_output = NULL, *d_output = NULL;
	h_output = (float *)malloc(num_bytes);
	cudaMalloc((void **)&d_output, num_bytes);

	//h_left_boundary sent to the left and h_right_halo is the space to receive that data
	//stencil data to be sent
	float *h_left_boundary = NULL, *h_right_boundary = NULL;
	//space to receive the stencil data
	float *h_left_halo = NULL, *h_right_halo = NULL;

	/* alloc host memory for halo data */
	cudaHostAlloc((void **)&h_left_boundary, num_halo_bytes, cudaHostAllocDefault);
	cudaHostAlloc((void **)&h_right_boundary, num_halo_bytes, cudaHostAllocDefault);
	cudaHostAlloc((void **)&h_left_halo, num_halo_bytes, cudaHostAllocDefault);
	cudaHostAlloc((void **)&h_right_halo, num_halo_bytes, cudaHostAllocDefault);

	/* create streams used for stencil computation */
	cudaStream_t stream0, stream1;
	cudaStreamCreate(&stream0);
	cudaStreamCreate(&stream1);
	
	int left_neighbor = (pid > 0) ? (pid - 1) : MPI_PROC_NULL;
	int right_neighbor = (pid < np -2) ? (pid + 1) : MPI_PROC_NULL;

	int left_halo_offset = 0;
	int right_halo_offset = dimx * (dimy + RADIO);

	//HPV checked
	int left_stage1_offset = num_halo_points;
	int right_stage1_offset = num_points - (2 * num_halo_points);

	int stage2_offset_ini = num_halo_points * RADIO * 2;
	int stage2_offset_fin = num_points - (num_halo_points * RADIO * 2);

/***
#ifdef DEBUG
	printf("pid: %d, MPI_PROC_NULL: %d\n", pid, MPI_PROC_NULL);
	printf("pid: %d, left_neighbor: %d\n", pid, left_neighbor); 
	printf("pid: %d, right_neighbor: %d\n", pid, right_neighbor);
	printf("pid: %d, right_halo_offset: %d\n", pid, right_halo_offset);
	printf("pid: %d, right_stage1_offset: %d\n", pid, right_stage1_offset);
	printf("pid: %d, stage2_offset: %d\n", pid, stage2_offset);
#endif
***/
	MPI_Barrier( MPI_COMM_WORLD);
/**************************************************************************************/
	for(int i=0; i < nreps; i++)
	{

		/* Compute boundary values needed by other nodes first */
		/* the first node does not compute left stencil neighbor */
		if (pid != 0)
			launch_kernel_edges(d_output, d_input, dimx, (dimy + (RADIO * 2) ), left_stage1_offset, RADIO, stream0);	
		/* the last node does not compute right stencil neighbor */
		if (pid != np-2)
			launch_kernel_edges(d_output, d_input, dimx, (dimy + (RADIO * 2) ), right_stage1_offset, RADIO, stream0);
	
		/* compute the remaining points */
		launch_kernel_internal(d_output, d_input, dimx, (dimy + (RADIO * 2) ), stage2_offset_ini, stage2_offset_fin, stream1);

		/* copy the data needed by other nodes to the host */
		cudaMemcpyAsync(h_left_boundary, d_output + left_stage1_offset, num_halo_bytes, cudaMemcpyDeviceToHost, stream0);
		cudaMemcpyAsync(h_right_boundary, d_output + right_stage1_offset, num_halo_bytes, cudaMemcpyDeviceToHost, stream0);
		cudaStreamSynchronize(stream0);

#ifdef DEBUG
	printf("pid: %d nreps: %d iteracion: %d, kernel================\n", pid, nreps, i);
        display_data(h_left_boundary, dimx, RADIO );
	display_data(h_right_boundary, dimx, RADIO );
#endif

		/* send data to left, get data from right */
		MPI_Sendrecv(h_left_boundary, num_halo_points, MPI_FLOAT, left_neighbor, i, h_right_halo, num_halo_points, MPI_FLOAT, right_neighbor, i, MPI_COMM_WORLD, &status);

		/* send data to right, get data from left */
		MPI_Sendrecv(h_right_boundary, num_halo_points, MPI_FLOAT, right_neighbor, i, h_left_halo, num_halo_points, MPI_FLOAT, left_neighbor, i, MPI_COMM_WORLD, &status);

		cudaMemcpyAsync(d_output+left_halo_offset, h_left_halo, num_halo_bytes, cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(d_output+right_halo_offset, h_right_halo, num_halo_bytes, cudaMemcpyHostToDevice, stream0);
		cudaDeviceSynchronize();

		float *temp  = d_output;
		d_output = d_input; d_input = temp;
	}
/***
#ifdef DEBUG
        printf("pid: %d Fin for kernel================\n", pid );
#endif
***/


	/* Wait for the previous communications */
	//MPI_Barrier(MPI_COMM_WORLD);
	
	float *temp = d_output;
	d_output = d_input;
	d_input = temp;

	/* send the output, skipping halo points */
	cudaMemcpy(h_output, d_output, num_bytes, cudaMemcpyDeviceToHost);
	float *send_address = h_output + num_halo_points;
	MPI_Send(send_address, dimx * dimy, MPI_REAL, server_process, DATA_COLLECT, MPI_COMM_WORLD);
	//MPI_Barrier(MPI_COMM_WORLD);

/***
#ifdef DEBUG
	printf("pid: %d FINAL================\n", pid );
	display_data(send_address, dimx, (dimy + RADIO *2) );
#endif
***/

	/* Release resources */
	free(h_input); free(h_output);
	cudaFreeHost(h_left_boundary); cudaFreeHost(h_right_boundary);
	cudaFreeHost(h_left_halo); cudaFreeHost(h_right_halo);
	cudaFree(d_input); cudaFree(d_output);
}


int main(int argc, char *argv[])
{
	//int pad = 0, dimx = 480 + pad, dimy = 480, dimz = 400, nreps = 100;
	int pad = 0, dimx = 9 + pad, dimy = 9, nreps = 3;
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
		//printf("compute_process node: %d\n", pid);
		compute_process(dimx, dimy/ (np - 1), nreps);
	else
		data_server( dimx, dimy);

	MPI_Finalize();
	return 0;
}

////////////////////////////////////////////////////////////////////////////////
//! HPV Random number generator
////////////////////////////////////////////////////////////////////////////////
#define TOP	2147483647.		// 2^31 - 1	
float Ranf( float low, float high )
{
	float r = (float)rand( );
	return(   low  +  r * ( high - low ) / (float)RAND_MAX   );
}


