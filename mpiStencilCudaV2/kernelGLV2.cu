// Device code
///////////////////////////////////////////////////////////////////////////////
//! Kernel to modify vertex positions 
//! @param data  data in global memory
///////////////////////////////////////////////////////////////////////////////


__global__ void kernel(float4 *agent, float4 *ids, int *d_world, int world_width, int world_height, int agent_width, int world_height_node, int pid)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	float PI = 3.14159265358979323846;
	int rebote = 0;
	int dist_prev = 1;
	int limit_width = world_width/6;
	int limit_height = world_height/6;


	if(ids[y*agent_width+x].y != 1 )//Verifica que el agente este activo
		ids[y*agent_width+x].y = -1;
	
	ids[y*agent_width+x].w += 1;

	//HPV movimiento de los agentes
	// 1ra opcion seguir en la misma direccion, se revisa el mundo para ver si esta disponible las sig coordenadas
	// en caso contrario se mueve 45 grados en sentido antihorario
	if(ids[y*agent_width+x].y == 1)//Verifica que el agente este activo
	{
		//Verificando si estas cerca de la orilla en la siguiente coordenada
		//next position = actual position + (cos(teta)*velocity)	
		int ccx = round( agent[y*agent_width+x].x + ( cos(agent[y*agent_width+x].z) * 2 * agent[y*agent_width+x].w) );
		int ccy = round( agent[y*agent_width+x].y + ( sin(agent[y*agent_width+x].z) * 2 * agent[y*agent_width+x].w) );
		if( ccx < limit_width || ccx > world_width - limit_width || ccy < limit_height || ccy > world_height - limit_height )
		{
			//si la siguiente coordenada sale del mundo entonces el angulo cambia 90 grados "rebote"
			agent[y*agent_width+x].z += PI/2;
			rebote = 1;
		}

		//calculando las coordenadas originales, marcas la coordenada original como disponible
		int cx_old = round( agent[y*agent_width+x].x ) ;
		int cy_old = ( round( agent[y*agent_width+x].y ) - 1 ) * world_width ;
		if ( cy_old < 0 )
			cy_old = 0 ;
		int coord_old =  cy_old + cx_old ;
	
		//Aqui revisas que la nueva posicion no este ocupada, si se trata de un rebote haces una excepcion y permites la colision
		// si esta ocupada la sig posicion te vas moviendo 45 grados en sentido antihorario
		// se utiliza world_width porque se hace la conversion a un arreglo dimensional de uno bidimensional
		int cx = round( agent[y*agent_width+x].x + ( cos(agent[y*agent_width+x].z) * dist_prev *agent[y*agent_width+x].w) ) ;
		int cy = ( round( agent[y*agent_width+x].y +  ( sin(agent[y*agent_width+x].z)  * dist_prev *agent[y*agent_width+x].w) ) - 1 )* world_width ;
		int coord =  cx + cy ;
		if( d_world[coord] == 0 || rebote )
		{
			agent[y*agent_width+x].x = agent[y*agent_width+x].x + ( cos(agent[y*agent_width+x].z) *agent[y*agent_width+x].w) ;
			agent[y*agent_width+x].y = agent[y*agent_width+x].y + ( sin(agent[y*agent_width+x].z) *agent[y*agent_width+x].w) ;
			d_world[coord] = ids[y*agent_width+x].x ;
			d_world[coord_old] = 0;
		}
		else{
			cx = round( agent[y*agent_width+x].x + ( cos(agent[y*agent_width+x].z + PI/4)  * dist_prev * agent[y*agent_width+x].w) ) ;
			cy = ( round( agent[y*agent_width+x].y + ( sin(agent[y*agent_width+x].z + PI/4)  * dist_prev *agent[y*agent_width+x].w) ) -1 ) * world_width ;
			coord =  cy + cx ;
			if( d_world[coord] == 0)
			{
				agent[y*agent_width+x].x = agent[y*agent_width+x].x + ( cos(agent[y*agent_width+x].z + PI/4) *agent[y*agent_width+x].w) ;
				agent[y*agent_width+x].y = agent[y*agent_width+x].y + ( sin(agent[y*agent_width+x].z + PI/4) *agent[y*agent_width+x].w) ;
				d_world[coord] = ids[y*agent_width+x].x ;
				d_world[coord_old] = 0;
			}
			else{
				cx = round( agent[y*agent_width+x].x + ( cos(agent[y*agent_width+x].z + 2*PI/4)  * dist_prev *agent[y*agent_width+x].w) ) ;
				cy = ( round( agent[y*agent_width+x].y + ( sin(agent[y*agent_width+x].z + 2*PI/4)  * dist_prev *agent[y*agent_width+x].w) ) - 1) * world_width;
				coord =  cy + cx ;
				if( d_world[coord] == 0)
				{
					agent[y*agent_width+x].x = agent[y*agent_width+x].x + ( cos(agent[y*agent_width+x].z + 2*PI/4) *agent[y*agent_width+x].w) ;
					agent[y*agent_width+x].y = agent[y*agent_width+x].y + ( sin(agent[y*agent_width+x].z + 2*PI/4) *agent[y*agent_width+x].w) ;
					d_world[coord] = ids[y*agent_width+x].x ;
					d_world[coord_old] = 0;
				}
				else{
					cx = round( agent[y*agent_width+x].x + ( cos(agent[y*agent_width+x].z + 3*PI/4)  * dist_prev *agent[y*agent_width+x].w) ) ;
					cy = ( round( agent[y*agent_width+x].y + ( sin(agent[y*agent_width+x].z + 3*PI/4) * dist_prev *agent[y*agent_width+x].w) ) -1 ) * world_width ;
					coord =  cy + cx ;
					if( d_world[coord] == 0)
					{
						agent[y*agent_width+x].x = agent[y*agent_width+x].x + ( cos(agent[y*agent_width+x].z + 3*PI/4) *agent[y*agent_width+x].w) ;
						agent[y*agent_width+x].y = agent[y*agent_width+x].y + ( sin(agent[y*agent_width+x].z + 3*PI/4) *agent[y*agent_width+x].w) ;
						d_world[coord] = ids[y*agent_width+x].x ;
						d_world[coord_old] = 0;
					}
					else{
						cx = round( agent[y*agent_width+x].x + ( cos(agent[y*agent_width+x].z + PI)  * dist_prev *agent[y*agent_width+x].w) ) ;
						cy = ( round( agent[y*agent_width+x].y + ( sin(agent[y*agent_width+x].z + PI) * dist_prev *agent[y*agent_width+x].w) ) -1 ) * world_width;
						coord =  cy + cx ;
						if( d_world[coord] == 0)
						{
							agent[y*agent_width+x].x = agent[y*agent_width+x].x + ( cos(agent[y*agent_width+x].z + PI) *agent[y*agent_width+x].w) ;	
							agent[y*agent_width+x].y = agent[y*agent_width+x].y + ( sin(agent[y*agent_width+x].z + PI) *agent[y*agent_width+x].w) ;
							d_world[coord] = ids[y*agent_width+x].x ;
							d_world[coord_old] = 0;
						}
						else{
							cx = round( agent[y*agent_width+x].x + ( cos(agent[y*agent_width+x].z + 5*PI/4)  * dist_prev *agent[y*agent_width+x].w) ) ;
							cy = ( round( agent[y*agent_width+x].y + ( sin(agent[y*agent_width+x].z + 5*PI/4)  * dist_prev *agent[y*agent_width+x].w) ) - 1 ) * world_width;
							coord =  cy + cx ;
							if( d_world[coord] == 0)
							{
								agent[y*agent_width+x].x = agent[y*agent_width+x].x + ( cos(agent[y*agent_width+x].z + 5*PI/4) *agent[y*agent_width+x].w) ;
								agent[y*agent_width+x].y = agent[y*agent_width+x].y + ( sin(agent[y*agent_width+x].z + 5*PI/4) *agent[y*agent_width+x].w) ;
								d_world[coord] = ids[y*agent_width+x].x ;
								d_world[coord_old] = 0;
							}
							else{
								cx = round( agent[y*agent_width+x].x + ( cos(agent[y*agent_width+x].z + 6*PI/4) * dist_prev *agent[y*agent_width+x].w) ) ;
								cy = ( round( agent[y*agent_width+x].y + ( sin(agent[y*agent_width+x].z + 6*PI/4) * dist_prev *agent[y*agent_width+x].w) ) -1 ) * world_width;
								coord =  cy + cx ;
								if( d_world[coord] == 0)
								{
									agent[y*agent_width+x].x = agent[y*agent_width+x].x + ( cos(agent[y*agent_width+x].z + 6*PI/4) *agent[y*agent_width+x].w) ;
									agent[y*agent_width+x].y = agent[y*agent_width+x].y + ( sin(agent[y*agent_width+x].z + 6*PI/4) *agent[y*agent_width+x].w) ;
									d_world[coord] = ids[y*agent_width+x].x ;
									d_world[coord_old] = 0;
								}
								else{
									cx = round( agent[y*agent_width+x].x + ( cos(agent[y*agent_width+x].z + 7*PI/4) * dist_prev *agent[y*agent_width+x].w) ) ;
									cy = ( round( agent[y*agent_width+x].y + ( sin(agent[y*agent_width+x].z + 7*PI/4) * dist_prev *agent[y*agent_width+x].w) ) - 1 ) * world_width;
									coord =  cy + cx ;
									if( d_world[coord] == 0)
									{
										agent[y*agent_width+x].x = agent[y*agent_width+x].x + ( cos(agent[y*agent_width+x].z + 7*PI/4) *agent[y*agent_width+x].w) ;
										agent[y*agent_width+x].y = agent[y*agent_width+x].y + ( sin(agent[y*agent_width+x].z + 7*PI/4) *agent[y*agent_width+x].w) ;
										d_world[coord] = ids[y*agent_width+x].x ;
										d_world[coord_old] = 0;
									}
									else{
										//si todas las posiciones a su alrededor estan ocupadas se queda donde esta y marcas
										//ocupada de nuevo esa posicion
										//d_world[coord_old] = 1;
		//si todas las posiciones a su alrededor estan ocupadas avanzas en la direccion original aunque se colisione
		int cx = round( agent[y*agent_width+x].x + ( cos(agent[y*agent_width+x].z) * dist_prev *agent[y*agent_width+x].w) ) ;
		int cy = ( round( agent[y*agent_width+x].y +  ( sin(agent[y*agent_width+x].z)  * dist_prev *agent[y*agent_width+x].w) ) - 1 )* world_width ;
		int coord =  cx + cy ;
						
		agent[y*agent_width+x].x = agent[y*agent_width+x].x + ( cos(agent[y*agent_width+x].z) *agent[y*agent_width+x].w) ;
		agent[y*agent_width+x].y = agent[y*agent_width+x].y + ( sin(agent[y*agent_width+x].z) *agent[y*agent_width+x].w) ;
		d_world[coord] = ids[y*agent_width+x].x ;
		d_world[coord_old] = 0;
	ids[y*agent_width+x].w = 9;
			
									}
								} //7*PI/4
							} //6*PI/4
						}//5*PI/4
					}//PI
				}//3*PI/4
			 }//PI/2
		}//PI/4

		//check if the agent should be computed by other node in the iteration according to its 'y' coordinate 
		if( round(agent[y*agent_width+x].y) < (pid * world_height_node) ) 
		{
			ids[y*agent_width+x].y = 2;
			d_world[coord_old] = 0; //queda vacia la casilla en el mundo
		}
		else if( round(agent[y*agent_width+x].y) > ( (pid + 1) * world_height_node ) )  
		{	
			ids[y*agent_width+x].y = 3;
			d_world[coord_old] = 0; //queda vacia la casilla en el mundo
		}
	}//if active
}



// CUDA computation on each node
// No MPI here, only CUDA
extern "C" void launch_kernel(float4 *d_agents_in, float4 *d_agents_ids, int *d_world, int world_width, int world_height, int agent_width, int agent_height, int world_height_node, int pid)
{
    // execute the kernel
    //dim3 block(agent_width, agent_height, 1);
    int block_width = 8;
    int block_height = 8;
    dim3 block(block_width, block_height, 1);	
    dim3 grid(agent_width / block.x + 1, agent_height / block.y + 1, 1);
//    dim3 grid(agent_width / block.x, mesh_height / block.y, 1);
    kernel<<< grid, block>>>(d_agents_in, d_agents_ids, d_world, world_width, world_height, agent_width, world_height_node, pid);  
}
/*
extern "C" void launch_kernel_init(float4* pos_ini, float4* d_world, unsigned int agent_width, unsigned int window_width)
{
    // execute the kernel
    dim3 block(8, 8, 1);
    dim3 grid(agent_width / block.x, agent_width / block.y, 1);
    kernel_init<<< grid, block>>>(pos_ini, d_world, agent_width, window_width);
}
*/
