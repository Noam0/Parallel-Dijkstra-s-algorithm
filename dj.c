#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <mpi.h>
#include <omp.h>

typedef unsigned int VERTEX;

#define MASTER 0
enum goal { FIND_ONE_DISTANCE, FIND_ALL_DISTANCES} goal;
enum tags { WORK, STOP };

struct vertex {
     VERTEX vertex; 
     unsigned int distance;
};

const unsigned int INFINITY = 1000000;

unsigned int *edges;  
int NV;   
int *done;
int  *distance;		  
VERTEX destination;
int local_start;
int local_end; 
						
void init(int argc, char **argv);
void doWork(int rank ,int nprocs);
struct vertex find_vertex_with_minimum_distance(int local_start, int local_end);
void update_distances(struct vertex current, int local_start, int local_end);
void printGraph();
void initAllArrays(int rank ,int nprocs);
MPI_Datatype getVertexType(struct vertex vertex);
int lineno = 1;
void skip_white_space();
void printDistances(char *s,double time);
void readGraph(void);


int main(int argc, char **argv)
{  

    MPI_Init(&argc, &argv);
    
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    
    double time; 
   if (rank == MASTER){
   	time = MPI_Wtime();
    init(argc,argv);
   } 
   
   initAllArrays(rank,nprocs);
   doWork(rank,nprocs);
   if (rank == MASTER){
	if (goal == FIND_ALL_DISTANCES)
        printDistances(NULL,time);
	else // goal == FIND_ONE_DISTANCE
     	if (distance[destination] == INFINITY)
            printf("no path to vertex %u\n", destination);			
		else printf("distance from 0 to %u is %u\n", destination, 
	            distance[destination]);
   }
   MPI_Finalize();
   return EXIT_SUCCESS;
}

void init(int argc, char **argv)
{ 
    readGraph();
    
    if (argc > 1) {
        goal = FIND_ONE_DISTANCE;
        destination = atoi(argv[1]);
		if (destination >= NV) {
			fprintf(stderr, "illegal destination vertex\n");
			exit(4);
		}
    } else
        goal = FIND_ALL_DISTANCES;		
}

void initAllArrays(int rank, int nprocs) {

MPI_Bcast(&NV, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

int jobSize = NV / (nprocs - 1);
local_start = (rank == MASTER) ? 0 : (rank - 1) * jobSize + 1;
local_end = (rank == nprocs - 1 || rank == MASTER) ? NV : local_start + jobSize;

distance = malloc((local_end-local_start)*sizeof(int));

done = malloc((local_end-local_start)*sizeof(int)); 


if (distance == NULL || done == NULL) {
 perror("malloc"); exit(1);}
 

for (VERTEX v = 0; v < local_end-local_start; v++)  {
        done[v] = 0;
        distance[v] = INFINITY;
    }
    if(rank == MASTER)
    	distance[0] = 0;
    if(rank!=MASTER)
    {
	edges = (unsigned int *)malloc(NV * NV * sizeof(unsigned int));
	if (edges == NULL) { perror("malloc"); exit(1);}
    }
MPI_Bcast(edges, NV*NV, MPI_UNSIGNED, MASTER, MPI_COMM_WORLD);
}




void doWork(int rank, int nprocs)
{ 
struct vertex current;
MPI_Datatype mpi_vertex_type = getVertexType(current);
int chunk = NV / (nprocs - 1);
int start = local_start;
int end = local_end;
MPI_Status status;
int local_current[2] = {INFINITY,INFINITY};
int temp_reduce_vertex[2] = {INFINITY,INFINITY};

   for (int step = 0; step < NV; step++)  {
      if (rank != 0){
			if (step == 0) {
				current.vertex = 0;
				current.distance = 0;
			}
			else {
				current = find_vertex_with_minimum_distance(local_start,local_end);
			}
			local_current[0] = current.distance;
			local_current[1] = current.vertex;
		}
		MPI_Reduce(local_current, temp_reduce_vertex, 1, MPI_2INT, MPI_MINLOC, MASTER, MPI_COMM_WORLD); //get the min of all processes 


		if (rank == MASTER){
			current.distance = temp_reduce_vertex[0];
			current.vertex = temp_reduce_vertex[1];
			if (current.distance >= INFINITY || goal == FIND_ONE_DISTANCE && current.vertex == destination){
				for (int i = 1; i < nprocs; i++)
					MPI_Send(NULL, 0, mpi_vertex_type, i, STOP, MPI_COMM_WORLD);
				break;
			}else
				for (int i = 1; i < nprocs; i++)
					MPI_Send(&current, 1, mpi_vertex_type, i, WORK, MPI_COMM_WORLD);
		}

		else{
			MPI_Recv(&current, 1, mpi_vertex_type, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			if (status.MPI_TAG == STOP)
				break;
			else{
				if(current.vertex >= start && current.vertex < end){
					done[current.vertex - start] = 1;
					}
				update_distances(current, local_start, local_end);
			}
		}
	}
	if (rank == 0)
	{
		for (int i = 1; i < nprocs; i++)
		{
			start = (i - 1) * chunk + 1;
			end = (i == nprocs - 1) ? NV : start + chunk;
			MPI_Recv(distance + start, end - start, MPI_INT, i, MASTER, MPI_COMM_WORLD, &status);
		}
	}

	else
	{
		MPI_Send(distance, end - start, MPI_INT, 0, MASTER, MPI_COMM_WORLD);
	}
	MPI_Type_free(&mpi_vertex_type);
} 

struct vertex find_vertex_with_minimum_distance(int local_start, int local_end){  
  
struct vertex vmin;
vmin.distance = INFINITY;

#pragma omp declare reduction(min_vertex:struct vertex : \
    omp_out = (omp_out.distance < omp_in.distance ? omp_out : omp_in)) initializer(omp_priv = omp_orig)
#pragma omp parallel for default(none) shared(done, distance,local_start, local_end) reduction(min_vertex:vmin)
for (VERTEX v = 0; v < local_end - local_start; v++) {
	if (!done[v] && distance[v] < vmin.distance) {
			vmin.distance = distance[v];
			vmin.vertex = v+local_start;
		}
	}
	return vmin;
}

void update_distances(struct vertex current, int local_start, int local_end){
#pragma omp parallel for default(none) shared(edges,done,distance,NV,current,local_start, local_end)
   for (VERTEX v = 0; v < local_end - local_start; v++) 
       if (!done[v]) {
           unsigned int alternative = current.distance + edges[current.vertex*NV+(v+local_start)];
           if (alternative < distance[v])
               distance[v] = alternative; 
       }
}



MPI_Datatype getVertexType(struct vertex vertex){
    MPI_Datatype mpi_vertex_type;
    int block_lengths[3] = {1, 1};
    MPI_Aint displacements[2] = {0};
    MPI_Aint base, addr_vertex, addr_distance;
    MPI_Get_address(&vertex, &base);
    MPI_Get_address(&vertex.vertex, &addr_vertex);
    MPI_Get_address(&vertex.distance, &addr_distance);
    displacements[0] = addr_vertex - base;
    displacements[1] = addr_distance - base;
    
    MPI_Datatype types[2] = {MPI_UNSIGNED, MPI_UNSIGNED};
    MPI_Type_create_struct(2, block_lengths, displacements, types, &mpi_vertex_type);
    MPI_Type_commit(&mpi_vertex_type);
	return mpi_vertex_type;
}

void readGraph() {
    
    int c;
    unsigned int w;
    int count_w = 0; 
        
    if (scanf("%d", &NV) == 1) {
         edges = (unsigned int *)malloc(NV * NV * sizeof(unsigned int));
         if (edges == NULL) { perror("malloc"); exit(1);}
    } else {
        fprintf(stderr, 
                "line %d: first item in the input should be the number of vertices in the graph\n",
                lineno);
        exit(1);
    }

    unsigned int *next_entry = edges;

    while (1) {
        skip_white_space();
        c = getchar();
        if (c == EOF) 
            break;
        if (count_w >= NV*NV) {
             fprintf(stderr, "line %d: too many weights (expecting %d*%d weights)\n",
                              lineno, NV, NV);
            exit(5);
        }
        if (c == '*') {
             *next_entry++ = INFINITY;
             count_w++;
        } else {
             ungetc(c, stdin);
             int r = scanf("%u", &w);
             if (r == 1) { // a number (weight) was read
                *next_entry++ = w;
                count_w++;
             } else {
                  fprintf(stderr, "line %d: error in input\n", lineno);
                  exit(2);
             }
        }
        
    }
    if (count_w != NV*NV) {
        fprintf(stderr, "%d weights appear in the input (expected\
 %d weights because number of vertices is %d)\n", 
         count_w, NV*NV, NV);
         exit(6);
    }
}
    
void skip_white_space() {
   int c;
   while(1) {
       if ((c = getchar()) == '\n')
           lineno++;
       else if (isspace(c))
           continue;
       else if (c == EOF)
           break;
       else {
         ungetc(c, stdin);
         break;
       }
   }
}

void printDistances(char *s , double time) { 
    
   if (s) printf("%s\n", s);
   for (VERTEX v = 0; v < NV; v++)
       if (distance[v] >= INFINITY)
           printf("%u:*\n", v);
       else
           printf("%u:%u\n", v, distance[v]);
           
    printf("Run time: %lf\n", MPI_Wtime() - time);       
}

void printGraph() {

    printf("graph weights:\n");
    for (int i = 0; i < NV; i++)  {
        for (int j = 0; j < NV; j++)
            if (edges[NV*i+j] >= INFINITY)
                printf("*  ");
            else 
                printf("%u  ", edges[NV*i+j]);
         putchar('\n');
     }
}

