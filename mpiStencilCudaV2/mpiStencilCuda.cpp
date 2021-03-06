#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <stddef.h>
#include <string.h>

// OpenGL Graphics includes
#include <GL/glew.h>
#include <GL/freeglut.h>

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <timer.h>               

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop

#include <vector_types.h>

//Compilation: make
//Local Execution: optirun mpirun -np 4 ./mpiStencilCuda
/*******************************************************************
usar git
envio de mensajes mpi dinamicos
uso de gpu direct
hacer pruebas en minotauro
partir el mundo en renglones y columnas
usar tiles en la gpu
version ompss

uso de git
*********************************************************************/



#define DATA_COLLECT 3
#define RADIO 1
//#define DEBUG
#define REFRESH_DELAY     10 //ms

#define agent_width 100
#define agent_height 100
#define world_width 1024
#define world_height 3072
#define nreps 15
int agents_total =  agent_width * agent_height;

struct sAgents{
	float4 pos[agent_width * agent_height];
	float4 ids[agent_width * agent_height];
};

/*****************************************************************/
/** Visualization						**/
/*****************************************************************/
// vbo variables
GLuint vbo;
struct cudaGraphicsResource *cuda_vbo_resource;
void *d_vbo_buffer = NULL;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = 0.0;

StopWatchInterface *timer = NULL;

// Auto-Verification Code
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
float avgFPS = 0.0f;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;

#define MAX(a,b) ((a > b) ? a : b)

void cleanup();

// GL functionality
bool initGL(int *argc, char **argv);
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
               unsigned int vbo_res_flags);
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res);

// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void timerEvent(int value);
void idle(void);

bool runDisplay(int argc, char **argv);
void refreshData(struct cudaGraphicsResource **vbo_resource);
void display_data(sAgents h_agents_in);
void computeFPS();

const char *sSDKsample = "Crowd (mpi+cuda+opengl)";

/*****************************************************************/
/** Visualization End						**/
/*****************************************************************/

extern "C" void launch_kernel(float4 *d_agents_pos, float4 *d_agents_ids, int *d_world, int world_width_a, int world_height_a, int agent_width_a, int agent_height_a, int world_height_node, int pid);  
	
float Ranf( float, float );

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
bool initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(world_width, world_height);
    glutCreateWindow("MPI Cuda GL Interop (VBO)");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMotionFunc(motion);
    glutTimerFunc(REFRESH_DELAY, timerEvent,0);

    // initialize necessary OpenGL extensions
    glewInit();

    if (! glewIsSupported("GL_VERSION_2_0 "))
    {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return false;
    }

    // default initialization
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, world_width, world_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    //gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 10.0);
    gluPerspective(60.0, (GLfloat)world_width / (GLfloat) world_height, 0.1, 7424.0);  //HPV se cambio de 10 a 1024 el ultimo campo ZFar
    glTranslated(-512, 1536, -7424); //HPV Se cambia la longitud de alcance de vision con ZFar y tmb el sistema de coordenadas


    SDK_CHECK_ERROR_GL();

    return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Run OpenGL
////////////////////////////////////////////////////////////////////////////////
bool runDisplay(int argc, char **argv)
{
    // Create the CUTIL timer
    sdkCreateTimer(&timer);

    // First initialize OpenGL context, so we can properly set the GL for CUDA.
    // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
    if (false == initGL(&argc, argv))
    {
        return false;
    }

    // register callbacks
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutIdleFunc(idle);

    // create VBO
    createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);

    // start rendering mainloop
    glutMainLoop();

    atexit(cleanup);
   
    return true;
}

void idle(void)
{
    refreshData(&cuda_vbo_resource);
    glutPostRedisplay();
}
////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void refreshData(struct cudaGraphicsResource **vbo_resource)
{
	int np; 
	MPI_Comm_size(MPI_COMM_WORLD, &np);
	MPI_Status status;

	int num_comp_nodes = np -1;
	unsigned int num_bytes = sizeof(sAgents);
	sAgents h_agents_in, h_agents_out[num_comp_nodes];
        size_t size_agents = agents_total * sizeof(float4);

        // map OpenGL buffer object for writing from CUDA
        float4 *d_agents;
        // Error code to check return values for CUDA calls
        cudaError_t err = cudaSuccess;

        checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
        size_t num_agents_bytes;
        checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_agents, &num_agents_bytes,
                                                         *vbo_resource));
		
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
				h_agents_in.pos[i].z = 0;
				h_agents_in.pos[i].w = 1;
			}
		}	
	}
		
	#ifdef DEBUG
        printf("Final Data:\n");	
	display_data(h_agents_in);
	#endif
		
	// Copy the host pointer memory to the device memory
	err = cudaMemcpy(d_agents, h_agents_in.pos, size_agents, cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
	{
	    fprintf(stderr, "Failed to copy pointer from host to device (error code %s)!\n", cudaGetErrorString(err));
	    exit(EXIT_FAILURE);
	}


	// unmap buffer object
	checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
	
	/* release resources */
	//free(&h_agents_in);
	//free(&h_agents_out); 
}

////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
               unsigned int vbo_res_flags)
{
    assert(vbo);

    // create buffer object
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);

    // initialize buffer object
    unsigned int size = agent_width * agent_height * 4 * sizeof(float);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

    SDK_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res)
{
    // unregister this buffer object with CUDA
    cudaGraphicsUnregisterResource(vbo_res);

    glBindBuffer(1, *vbo);
    glDeleteBuffers(1, vbo);

    *vbo = 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
    sdkStartTimer(&timer);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);

    // render from the vbo
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(4, GL_FLOAT, 0, 0); 

    glEnableClientState(GL_VERTEX_ARRAY);
    glColor3f(1.0, 0.0, 0.0);
    glPointSize(6);
    glDrawArrays(GL_POINTS, 0, agent_width * agent_height);
    glDisableClientState(GL_VERTEX_ARRAY);

    glColor3f(0.0,1.0,0.0) ;

    //Begin drawing lines
    glBegin(GL_LINES); 
    // 2 end points on line
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, world_height, 0.0f);
    glEnd();

    glBegin(GL_LINES); 
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(world_width, 0.0f, 0.0f);
    glEnd();

    glBegin(GL_LINES); 
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, world_width);
    glEnd();

    glBegin(GL_LINES); 
    glVertex3f(0.0f, world_height, 0.0f);
    glVertex3f(world_width, world_height, 0.0f);
    glEnd();

    glBegin(GL_LINES); 
    glVertex3f(world_width, 0.0f, 0.0f);
    glVertex3f(world_width, world_height, 0.0f);
    glEnd();

    glBegin(GL_LINES); 
    glVertex3f(0.0f, world_height/3, 0.0f);
    glVertex3f(world_width, world_height/3, 0.0f);
    glEnd();

    glBegin(GL_LINES); 
    glVertex3f(0.0f, 2*(world_height/3), 0.0f);
    glVertex3f(world_width, 2*(world_height/3), 0.0f);
    glEnd();

    glBegin(GL_LINES); 
    glVertex3f(world_width/3, 0.0f, 0.0f);
    glVertex3f(world_width/3, world_height, 0.0f);
    glEnd();

    glBegin(GL_LINES); 
    glVertex3f(2*(world_width/3), 0.0f, 0.0f);
    glVertex3f(2*(world_width/3), world_height, 0.0f);
    glEnd();

    glutSwapBuffers();

    sdkStopTimer(&timer);
    computeFPS();
}

void timerEvent(int value)
{
    glutPostRedisplay();
    glutTimerFunc(REFRESH_DELAY, timerEvent,0);
}

void cleanup()
{
    sdkDeleteTimer(&timer);

    if (vbo)
    {
        deleteVBO(&vbo, cuda_vbo_resource);
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key)
    {
        case (27) :
            exit(EXIT_SUCCESS);
            break;
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        mouse_buttons |= 1<<button;
    }
    else if (state == GLUT_UP)
    {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - mouse_old_x);
    dy = (float)(y - mouse_old_y);

    if (mouse_buttons & 1)
    {
        rotate_x += dy * 0.2f;
        rotate_y += dx * 0.2f;
    }
    else if (mouse_buttons & 4)
    {
        translate_z += dy * 0.01f;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}


// Initialize agents with random positions
void init_data(sAgents &h_agents_in)
{
	const float PI = 3.1415926535;
	const float XMIN = 	{ world_width/3 };
	const float XMAX = 	{  2*world_width/3 };
	const float YMIN = 	{ world_height/3 };
	const float YMAX = 	{  2*world_height/3 };
	const float THMIN =	{   0.0};
	const float THMAX =	{    2*PI }; //2*PI
	const float VMIN =	{   1.0 }; //0.0
	const float VMAX =	{    2.0 };

	for(int i = 0; i< agents_total;i++)
	{
		h_agents_in.ids[i].x = i + 1;
		h_agents_in.ids[i].y = -1;
		h_agents_in.ids[i].z = -1;
		h_agents_in.ids[i].w = -1;
		h_agents_in.pos[i].x = round(Ranf( XMIN, XMAX ));
		h_agents_in.pos[i].y = round(Ranf( YMIN, YMAX ));
		h_agents_in.pos[i].z = Ranf( THMIN, THMAX );
		h_agents_in.pos[i].w = Ranf( VMIN, VMAX );
	}	
}

void display_data(sAgents h_agents_in)
{
    for( int i = 0; i < agents_total; i++ )
	printf("id: %f pid: %f status: %f flagMisc: %f x: %f y: %f z: %f, w: %f \n", h_agents_in.ids[i].x, h_agents_in.ids[i].z, h_agents_in.ids[i].y, h_agents_in.ids[i].w, h_agents_in.pos[i].x, h_agents_in.pos[i].y, h_agents_in.pos[i].z, h_agents_in.pos[i].w);
}


void data_server(int argc, char *argv[])
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
		
	runDisplay(argc, argv);
	
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
	}

	err = cudaMemcpy(d_world, h_world, size_world, cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
    	{
        	fprintf(stderr, "Failed to copy pointer from host to device (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}


	while(1)
	{
		int it=4;
		err = cudaMemcpy(d_agents_pos, h_agents_in.pos, num_bytes_agents, cudaMemcpyHostToDevice);

		if (err != cudaSuccess)
	    	{
	        	fprintf(stderr, "Failed to copy pointer from host to device (error code %s)!\n", cudaGetErrorString(err));
	        	exit(EXIT_FAILURE);
	    	}
	
		err = cudaMemcpy(d_agents_ids, h_agents_in.ids, num_bytes_agents, cudaMemcpyHostToDevice);

		if (err != cudaSuccess)
	    	{
	        	fprintf(stderr, "Failed to copy pointer from host to device (error code %s)!\n", cudaGetErrorString(err));
	        	exit(EXIT_FAILURE);
	    	}


		launch_kernel(d_agents_pos, d_agents_ids, d_world, world_width, world_height, agent_width, agent_height, world_height_node, pid );

		cudaMemcpy(h_agents_in.pos, d_agents_pos, num_bytes_agents, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_agents_in.ids, d_agents_ids, num_bytes_agents, cudaMemcpyDeviceToHost);

		MPI_Barrier(MPI_COMM_WORLD);

		#ifdef DEBUG
		//printf("After kernel pid: %d\n", pid);
		//display_data(h_agents_in);
		#endif

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
					h_agents_in.ids[i].y = 1;
					h_agents_in.ids[i].z = pid;						
				}
			}
			if(pid != 0)				
			{
				if(h_agents_left_node.ids[i].y == 3)
				{
					h_agents_in.pos[i] = h_agents_left_node.pos[i];
					h_agents_in.ids[i].y = 1;
					h_agents_in.ids[i].z = pid;						
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

		MPI_Send(&h_agents_in, num_bytes, MPI_BYTE, server_process, DATA_COLLECT, MPI_COMM_WORLD);

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

/*************************************************************************/
/** Main Function							**/
/*************************************************************************/
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
		data_server(argc, argv);

	MPI_Finalize();
	return 0;
}

void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        avgFPS = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        fpsCount = 0;
        fpsLimit = (int)MAX(avgFPS, 1.f);

        sdkResetTimer(&timer);
    }

    char fps[256];
    sprintf(fps, "MPI Cuda GL Interop (VBO): %3.1f fps (Max 100Hz)", avgFPS);
    glutSetWindowTitle(fps);
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

