//  Matrix-Matrix Multiplication

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include<omp.h>
//using namespace std;
#define N 4

int A[N][N], B[N][N], C[N][N]; // declaring matrices of NxN size
int main ()
{
	/* DECLARING VARIABLES */
	int i, j, m; // indices for matrix multiplication
	float t_1; // Execution time measures
	clock_t c_1, c_2;
	/* FILLING MATRICES WITH RANDOM NUMBERS */
	for(i=0;i<N;i++) 
	{
		for(j=0;j<N;j++)
				 {
					A[i][j]= (rand()%5);
					B[i][j]= (rand()%5);
				 }
	}

       // Display input matrix A: 
	printf("Matrix A:\n");
	for(i=0;i<N;i++) 
	{
		for(j=0;j<N;j++)
				 {
					printf("%d\t",A[i][j]);
				 }
		printf("\n");
	}


       // Display input matrix B: 
	printf("Matrix B:\n");
	for(i=0;i<N;i++) 
	{
		for(j=0;j<N;j++)
				 {
					printf("%d\t",B[i][j]);
				 }
		printf("\n");
	}

	c_1=clock();  // time measure: 
	/* MATRIX MULTIPLICATION */
	printf("Max number of threads: %i \n",omp_get_max_threads());
	
	#pragma omp parallel
	#pragma omp single
		{
		printf("Number of threads: %i \n",omp_get_num_threads());
		}
	
	#pragma omp parallel for private(m,j)
	// #pragma omp_set_num_threads(8)
	for(i=0;i<N;i++) 
	{
				for(j=0;j<N;j++) 
				{
					C[i][j]=0.; // set initial value of resulting matrix C = 0
					
					for(m=0;m<N;m++) 
						{
							C[i][j]=A[i][m]*B[m][j]+C[i][j];
						}
	    		//	printf("C: %d \t",C[i][j]);
				}
			//	printf("\n");

	}


// Display input matrix B: 
	printf("Matrix C:\n");
	for(i=0;i<N;i++) 
	{
		for(j=0;j<N;j++)
				 {
					printf("%d\t",C[i][j]);
				 }
		printf("\n");
	}




	/* TIME MEASURE + OUTPUT */
	c_2=clock();  // time measure: 
	t_1 = (float)(c_2-c_1)/CLOCKS_PER_SEC; // in seconds; - time elapsed for job row-wise
	printf("Execution time: %f(in seconds) \n",t_1);
	

	/* TERMINATE PROGRAM */
	return 0;
}

/*


studen@student-ThinkCentre-M72e:~/HPC$ gcc -fopenmp matrix_matrix_multiplication.c -o mm
studen@student-ThinkCentre-M72e:~/HPC$ ./mm
Matrix A:
3	2	3	1	
4	2	0	3	
0	2	1	2	
2	2	2	4	
Matrix B:
1	0	0	2	
1	2	4	1	
1	1	3	4	
0	3	0	2	
Max number of threads: 4 
Number of threads: 4 
Matrix C:
8	10	17	22	
6	13	8	16	
3	11	11	10	
6	18	14	22	
Execution time: 0.029086(in seconds) 


*/
