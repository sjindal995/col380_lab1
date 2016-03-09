/* 
 * Purpose:  Use iterative depth-first search and OpenMP to solve an 
 *           instance of the travelling salesman problem.  
 *
 * Compile:  gcc -O3 -Wall -fopenmp -o tsp tsp.c
 * Usage:    ./tsp <thread count> <matrix_file>
 *
 * Input:    From a user-specified file, the number of cities
 *           followed by the costs of travelling between the
 *           cities organized as a matrix:  the cost of
 *           travelling from city i to city j is the ij entry.
 *           Costs are nonnegative ints.  Diagonal entries are 0.
 * Output:   The best tour found by the program and the cost
 *           of the tour.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <limits.h>


const int INFINITY = INT_MAX;
const int NO_CITY = -1;
const int FALSE = 0;
const int TRUE = 1;
const int MAX_STRING = 1000;

typedef int city_t;
typedef int cost_t;

typedef struct {
   city_t* cities; /* Cities in partial tour           */
   int count;      /* Number of cities in partial tour */
   cost_t cost;    /* Cost of partial tour             */
} tour_struct;

typedef tour_struct* tour_t;
#define City_count(tour) (tour->count)
#define Tour_cost(tour) (tour->cost)
#define Last_city(tour) (tour->cities[(tour->count)-1])
#define Tour_city(tour,i) (tour->cities[(i)])

typedef struct {
   tour_t* list;
   int list_sz;
   int list_alloc;
}  stack_struct;
typedef stack_struct* my_stack_t;

typedef struct {
   tour_t* list;
   int list_sz;
   int list_alloc;
   int start;
}  queue_struct;
typedef queue_struct* my_queue_t;

/* Global Vars: */
int n;  /* Number of cities in the problem */
int thread_count;
cost_t* digraph;
#define Cost(city1, city2) (digraph[city1*n + city2])
city_t home_town = 0;
tour_t best_tour;
int init_tour_count;

void Usage(char* prog_name);
void Read_digraph(FILE* digraph_file);
void Print_digraph(void);

void Par_tree_search(void); // TODO: Implement this function

void Set_init_tours(int my_rank, int* my_first_tour_p,
      int* my_last_tour_p);

void Print_tour(int my_rank, tour_t tour, char* title);
int  Best_tour(tour_t tour); 
void Update_best_tour(tour_t tour);
void Copy_tour(tour_t tour1, tour_t tour2);
void Add_city(tour_t tour, city_t);
void Remove_last_city(tour_t tour);
int  Feasible(tour_t tour, city_t city);
int  Visited(tour_t tour, city_t city);
void Init_tour(tour_t tour, cost_t cost);
tour_t Alloc_tour(my_stack_t avail);
void Free_tour(tour_t tour, my_stack_t avail);

void Init_stack(my_stack_t avail);
void Push(my_stack_t avail, tour_t tour);
void Push_copy(my_stack_t avail, tour_t tour);
tour_t Pop(my_stack_t avail);
int Empty_stack(my_stack_t avail);

void Init_queue(my_queue_t q_avail);
void qPush(my_queue_t q_avail, tour_t tour);
void qPush_copy(my_queue_t q_avail, tour_t tour);
tour_t qPop(my_queue_t q_avail);
int Empty_queue(my_queue_t q_avail);
void threadSearch(my_stack_t avail);

void Ser_tree_search(void);

/*------------------------------------------------------------------*/
int main(int argc, char* argv[]) {
   FILE* digraph_file;
   double start, finish;

   if (argc != 3) Usage(argv[0]);
   thread_count = strtol(argv[1], NULL, 10);
   if (thread_count <= 0) {
      fprintf(stderr, "Thread count must be positive\n");
      Usage(argv[0]);
   }
   digraph_file = fopen(argv[2], "r");
   if (digraph_file == NULL) {
      fprintf(stderr, "Can't open %s\n", argv[2]);
      Usage(argv[0]);
   }
   Read_digraph(digraph_file);
   fclose(digraph_file);
#  ifdef DEBUG
   Print_digraph();
#  endif   

   best_tour = Alloc_tour(NULL);
   Init_tour(best_tour, INFINITY);
#  ifdef DEBUG
   Print_tour(-1, best_tour, "Best tour");
   printf("City count = %d\n",  City_count(best_tour));
   printf("Cost = %d\n\n", Tour_cost(best_tour));
#  endif

   start = omp_get_wtime();
   Par_tree_search();
   // Ser_tree_search();
   /*
    * TODO: Implement the parallel tsp 
    * Par_tree_search(); 
   */
   finish = omp_get_wtime();
   
   Print_tour(-1, best_tour, "Best tour");
   printf("Cost = %d\n", best_tour->cost);
   printf("Time = %e seconds\n", finish-start);

   free(best_tour->cities);
   free(best_tour);
   free(digraph);
   return 0;
}  /* main */

/*------------------------------------------------------------------
 * Function:  Init_tour
 * Purpose:   Initialize the data members of allocated tour
 * In args:   
 *    cost:   initial cost of tour
 * Global in:
 *    n:      number of cities in TSP
 * Out arg:   
 *    tour
 */
void Init_tour(tour_t tour, cost_t cost) {
   int i;

   tour->cities[0] = 0; // hometown added as a starting point
   for (i = 1; i <= n; i++) {
      tour->cities[i] = NO_CITY;
   }
   tour->cost = cost;
   tour->count = 1;
}  /* Init_tour */


/*------------------------------------------------------------------
 * Function:  Usage
 * Purpose:   Inform user how to start program and exit
 * In arg:    prog_name
 */
void Usage(char* prog_name) {
   fprintf(stderr, "usage: %s <thread_count> <digraph file>\n", prog_name);
   exit(0);
}  /* Usage */

/*------------------------------------------------------------------
 * Function:  Read_digraph
 * Purpose:   Read in the number of cities and the digraph of costs
 * In arg:    digraph_file
 * Globals out:
 *    n:        the number of cities
 *    digraph:  the matrix file
 */
void Read_digraph(FILE* digraph_file) {
   int i, j;

   fscanf(digraph_file, "%d", &n);
   if (n <= 0) {
      fprintf(stderr, "Number of vertices in digraph must be positive\n");
      exit(-1);
   }
   digraph = malloc(n*n*sizeof(cost_t));

   for (i = 0; i < n; i++)
      for (j = 0; j < n; j++) {
         fscanf(digraph_file, "%d", &digraph[i*n + j]);
         if (i == j && digraph[i*n + j] != 0) {
            fprintf(stderr, "Diagonal entries must be zero\n");
            exit(-1);
         } else if (i != j && digraph[i*n + j] <= 0) {
            fprintf(stderr, "Off-diagonal entries must be positive\n");
            fprintf(stderr, "diagraph[%d,%d] = %d\n", i, j, digraph[i*n+j]);
            exit(-1);
         }
      }
}  /* Read_digraph */


/*------------------------------------------------------------------
 * Function:  Print_digraph
 * Purpose:   Print the number of cities and the digraphrix of costs
 * Globals in:
 *    n:        number of cities
 *    digraph:  digraph of costs
 */
void Print_digraph(void) {
   int i, j;

   printf("Order = %d\n", n);
   printf("Matrix = \n");
   for (i = 0; i < n; i++) {
      for (j = 0; j < n; j++)
         printf("%2d ", digraph[i*n+j]);
      printf("\n");
   }
   printf("\n");
}  /* Print_digraph */


/*------------------------------------------------------------------
 * Function:    Best_tour
 * Purpose:     Determine whether addition of the hometown to the 
 *              n-city input tour will lead to a best tour.
 * In arg:
 *    tour:     tour visiting all n cities
 * Ret val:
 *    TRUE if best tour, FALSE otherwise
 */
int Best_tour(tour_t tour) {
   cost_t cost_so_far = Tour_cost(tour);
   city_t last_city = Last_city(tour);

   if (cost_so_far + Cost(last_city, home_town) < Tour_cost(best_tour))
      return TRUE;
   else
      return FALSE;
}  /* Best_tour */


/*------------------------------------------------------------------
 * Function:    Update_best_tour
 * Purpose:     Replace the existing best tour with the input tour +
 *              hometown
 * In arg:
 *    tour:     tour that's visited all n-cities
 * Global out:
 *    best_tour:  the current best tour
 */
void Update_best_tour(tour_t tour) {

   if (Best_tour(tour)) {
      Copy_tour(tour, best_tour);
      Add_city(best_tour, home_town);
   }
}  /* Update_best_tour */


/*------------------------------------------------------------------
 * Function:   Copy_tour
 * Purpose:    Copy tour1 into tour2
 * In arg:
 *    tour1
 * Out arg:
 *    tour2
 */
void Copy_tour(tour_t tour1, tour_t tour2) {

   memcpy(tour2->cities, tour1->cities, (n+1)*sizeof(city_t));
   tour2->count = tour1->count;
   tour2->cost = tour1->cost;
}  /* Copy_tour */

/*------------------------------------------------------------------
 * Function:  Add_city
 * Purpose:   Add city to the end of tour
 * In arg:
 *    city
 * In/out arg:
 *    tour
 * Note: This should only be called if tour->count >= 1.
 */
void Add_city(tour_t tour, city_t new_city) {
   city_t old_last_city = Last_city(tour);
   tour->cities[tour->count] = new_city;
   (tour->count)++;
   tour->cost += Cost(old_last_city,new_city);
}  /* Add_city */

/*------------------------------------------------------------------
 * Function:  Remove_last_city
 * Purpose:   Remove last city from end of tour
 * In/out arg:
 *    tour
 * Note:
 *    Function assumes there are at least two cities on the tour --
 *    i.e., the hometown in tour->cities[0] won't be removed.
 */
void Remove_last_city(tour_t tour) {
   city_t old_last_city = Last_city(tour);
   city_t new_last_city;
   
   tour->cities[tour->count-1] = NO_CITY;
   (tour->count)--;
   new_last_city = Last_city(tour);
   tour->cost -= Cost(new_last_city,old_last_city);
}  /* Remove_last_city */

/*------------------------------------------------------------------
 * Function:  Feasible
 * Purpose:   Check whether nbr could possibly lead to a better
 *            solution if it is added to the current tour.  The
 *            function checks whether nbr has already been visited
 *            in the current tour, and, if not, whether adding the
 *            edge from the current city to nbr will result in
 *            a cost less than the current best cost.
 * In args:   All
 * Global in:
 *    best_tour
 * Return:    TRUE if the nbr can be added to the current tour.
 *            FALSE otherwise
 */
int Feasible(tour_t tour, city_t city) {
   city_t last_city = Last_city(tour);

   if (!Visited(tour, city) && 
        Tour_cost(tour) + Cost(last_city,city) < Tour_cost(best_tour))
      return TRUE;
   else
      return FALSE;
}  /* Feasible */


/*------------------------------------------------------------------
 * Function:   Visited
 * Purpose:    Use linear search to determine whether city has already
 *             been visited on the current tour.
 * In args:    All
 * Return val: TRUE if city has already been visited.
 *             FALSE otherwise
 */
int Visited(tour_t tour, city_t city) {
   int i;

   for (i = 0; i < City_count(tour); i++)
      if ( Tour_city(tour,i) == city ) return TRUE;
   return FALSE;
}  /* Visited */


/*------------------------------------------------------------------
 * Function:  Print_tour
 * Purpose:   Print a tour
 * In args:   All
 * Notes:      
 * 1.  Copying the tour to a string makes it less likely that the 
 *     output will be broken up by another process/thread
 * 2.  Passing a negative value for my_rank will cause the rank
 *     to be omitted from the output
 */
void Print_tour(int my_rank, tour_t tour, char* title) {
   int i;
   char string[MAX_STRING];

   if (my_rank >= 0)
      sprintf(string, "Th %d > %s %p: ", my_rank, title, tour);
   else
      sprintf(string, "%s = ", title);
   for (i = 0; i < City_count(tour); i++)
      sprintf(string + strlen(string), "%d ", Tour_city(tour,i));
   printf("%s\n", string);
}  /* Print_tour */


/*------------------------------------------------------------------
 * Function:  Alloc_tour
 * Purpose:   Allocate memory for a tour and its members
 * In/out arg:
 *    avail:  stack storing unused tours
 * Global in: n, number of cities
 * Ret val:   Pointer to a tour_struct with storage allocated for its
 *            members
 */
tour_t Alloc_tour(my_stack_t avail) {
   tour_t tmp;

   if (avail == NULL || Empty_stack(avail)) {
      // printf("entry\n");
      tmp = malloc(sizeof(tour_struct));
      // printf("exit\n");
      tmp->cities = malloc((n+1)*sizeof(city_t));
      return tmp;
   } else {
      return Pop(avail);
   }
}  /* Alloc_tour */

/*------------------------------------------------------------------
 * Function:  Free_tour
 * Purpose:   Free a tour
 * In/out arg:
 *    avail
 * Out arg:   
 *    tour
 */
void Free_tour(tour_t tour, my_stack_t avail) {
   if (avail == NULL) {
      free(tour->cities);
      free(tour);
   } else {
      Push(avail, tour);
   }
}  /* Free_tour */


void Init_stack(my_stack_t avail){
   avail->list = malloc(n*n*sizeof(tour_t));
   avail->list_sz = 0;
   avail->list_alloc = n*n;
}

void Push_copy(my_stack_t avail, tour_t tour){
   tour_t new_tour = Alloc_tour(NULL);
   Copy_tour(tour, new_tour);
   Push(avail,new_tour);
}

void Push(my_stack_t avail, tour_t tour){
   if(avail->list_sz == avail->list_alloc){
      printf("cannot be pushed. No more space on stack.\n");
      return;
   }
   avail->list[avail->list_sz] = tour;
   avail->list_sz++;
}

tour_t Pop(my_stack_t avail){
   if(Empty_stack(avail)){
      printf("cannot pop from empty stack.\n");
      return NULL;
   }
   avail->list_sz--;
   tour_t top = avail->list[avail->list_sz];
   avail->list[avail->list_sz] = NULL;
   return top;
}

int Empty_stack(my_stack_t avail){
   return (avail->list_sz == 0);
}

void Init_queue(my_queue_t q_avail){
   q_avail->list = malloc((thread_count+n)*sizeof(tour_t));
   q_avail->list_sz = 0;
   q_avail->list_alloc = thread_count+n;
   q_avail->start = 0;
}

void qPush_copy(my_queue_t q_avail, tour_t tour){
   tour_t new_tour = Alloc_tour(NULL);
   Copy_tour(tour, new_tour);
   qPush(q_avail,new_tour);
}

void qPush(my_queue_t q_avail, tour_t tour){
   if(q_avail->list_sz == q_avail->list_alloc){
      // printf("cannot be pushed. No more space on queue.\n");
      return;
   }
   q_avail->list[(q_avail->start + q_avail->list_sz)%(q_avail->list_alloc)] = tour;
   q_avail->list_sz++;
}

tour_t qPop(my_queue_t q_avail){
   if(Empty_queue(q_avail)){
      // printf("cannot pop from empty queue.\n");
      return NULL;
   }
   q_avail->list_sz--;
   tour_t front = q_avail->list[q_avail->start];
   q_avail->list[q_avail->start] = NULL;
   q_avail->start = (q_avail->start+1)%(q_avail->list_alloc);
   return front;
}

int Empty_queue(my_queue_t q_avail){
   return (q_avail->list_sz == 0);
}

void Par_tree_search(void){
   my_queue_t q_avail = malloc(sizeof(queue_struct));
   Init_queue(q_avail);
   tour_t tour = Alloc_tour(NULL);
   Init_tour(tour,0);
   qPush_copy(q_avail,tour);
   tour_t curr_tour;
   // printf("--------------------q_avail size: %d------------------- thread_count: %d\n",q_avail->list_sz, thread_count);
   while(!Empty_queue(q_avail) && (q_avail->list_sz < thread_count)){
      curr_tour = qPop(q_avail);
      if(City_count(curr_tour) == n){
         if(Best_tour(curr_tour)){
            Update_best_tour(curr_tour);
         }
      }  
      else{
         int nbr;
         for(nbr = n-1; nbr >= 1; nbr--){
            if(Feasible(curr_tour, nbr)){
               Add_city(curr_tour,nbr);
               qPush_copy(q_avail, curr_tour);
               Remove_last_city(curr_tour);
            }
         }
      }
      Free_tour(curr_tour,NULL);
   }
   if(Empty_queue(q_avail)){
      return;
   }
   my_stack_t* avail = malloc(thread_count*sizeof(my_stack_t));
   int i;
   for(i=0;i<thread_count;i++){
      avail[i] = malloc(sizeof(stack_struct));
      Init_stack(avail[i]);
   }
   for(i=0;i<q_avail->list_sz;i++){
      tour_t tour1 = q_avail->list[(q_avail->start + i)%(q_avail->list_alloc)];
      Push_copy(avail[i%thread_count],tour1);
   }
   #pragma omp parallel num_threads(thread_count)
   {
      int tid = omp_get_thread_num();
      threadSearch(avail[tid]);
   }
}

void threadSearch(my_stack_t avail){
   tour_t curr_tour;
   while(!Empty_stack(avail)){
      curr_tour = Pop(avail);
      if(City_count(curr_tour) == n){
         #pragma omp critical
         {
            if(Best_tour(curr_tour)){
               Update_best_tour(curr_tour);
            }
         }
      }  
      else{
         int nbr;
         for(nbr = n-1; nbr >= 1; nbr--){
            if(Feasible(curr_tour, nbr)){
               Add_city(curr_tour,nbr);
               Push_copy(avail, curr_tour);
               Remove_last_city(curr_tour);
            }
         }
      }
      Free_tour(curr_tour,NULL);
   }
}

void Ser_tree_search(){
   my_stack_t avail = malloc(sizeof(stack_struct));
   Init_stack(avail);
   tour_t tour = Alloc_tour(NULL);
   Init_tour(tour,0);
   Push_copy(avail,tour);
   while(!Empty_stack(avail)){
      tour_t curr_tour = Pop(avail);
      if(City_count(curr_tour) == n){
         if(Best_tour(curr_tour)){
            Update_best_tour(curr_tour);
         }
      }  
      else{
         int nbr;
         for(nbr = n-1; nbr >= 1; nbr--){
            if(Feasible(curr_tour, nbr)){
               Add_city(curr_tour,nbr);
               Push_copy(avail, curr_tour);
               Remove_last_city(curr_tour);
            }
         }
      }
   }
}