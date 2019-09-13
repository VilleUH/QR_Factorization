#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "functions.h"
#include "gram_schmidt.h"

int main(int argc, char *argv[]) {
	int n, m, size, rank, i, j, k, k_scatter, rest;
	double *A, *Q, *Q_vector, *R, *R_vector, *Q_recv, *QR_multiply, t;
	MPI_Comm extra_scatter;
	
	if (argc != 3) {
		printf("Correct usage \"mpirun -np n_processes ./qr_fac m n\"\n");
		return -1;
	}
	
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	/* m = number of rows, n = number of columns */
	m = atoi(argv[1]);
	n = atoi(argv[2]);
	
	if (rank == 0)
		printf("Starting QR-factorization with m = %d, n = %d and n_processes = %d\n...\n", m, n, size);
	
	/* Allocate memory and generate random input matrix A */
	if (rank == 0) {
		A = (double *)malloc(m*n*sizeof(double));
		Q = (double *)malloc(m*n*sizeof(double));
		R = (double *)malloc(n*n*sizeof(double));
		QR_multiply = (double *)malloc(m*n*sizeof(double));
		for (i = 0; i < m; i++) {
			for (j = 0; j < n; j++) {
				// A[i*n+j] = drand48();
				A[j*m+i] = (double)((int)(drand48()*10));
				Q[j*m+i] = A[j*m+i];
			}
		}
		/* Print input matrix A */
/* 		printf("\nA:\n");
		for (i = 0; i < m; i++) {
			for (j = 0; j < n; j++) {
				printf("%f  ", A[j*m+i]);
			}
			printf("\n");
		} */
	}
	
	/* if n_processes = 1 */
	if (size == 1) {
		t = MPI_Wtime();
		for (i = 0; i < n; i++) {
			R[i+i*n] = sqrt(dot_product(&Q[i*m], &Q[i*m], m));
			normalize(&Q[i*m], R[i+i*n], m);
			for (j = i+1; j < n; j++) {
				R[i+j*n] = dot_product(&Q[i*m], &Q[j*m], m);
				projection(&Q[j*m], R[i+j*n], &Q[i*m], m);
			}
		}
		t = MPI_Wtime() - t;
		printf("Wall time: %1.6f\n", t);
	} 
	else 
		
	/* if n_processes > 1 */
	{
		/* wrap-around number k */
		k = n/size;
		k_scatter = k;
		
		/* To handle the case where n%size != 0 a new communicator is 
		   created to partition the remaining vectors */
		rest = n%size;
		if (rank < rest)
			k++;
		MPI_Comm_split(MPI_COMM_WORLD, rank < rest, rank, &extra_scatter);
		
		/* Allocate memory to be used for the partitioning and Gram-Schmidt orthogonalization */
		Q_vector = (double *)malloc(k*m*sizeof(double));
		Q_recv = (double *)malloc(m*sizeof(double));
		R_vector = (double *)calloc(k*n,sizeof(double));
		
		if (rank == 0)
			t = MPI_Wtime();
		
		/* Scatter the columns of A between the processes */
		for (i = 0; i < k_scatter; i++) {
			MPI_Scatter(&A[i*size*m], m, MPI_DOUBLE, &Q_vector[i*m], m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		}
		
		/* Scatter the extra columns in the case where n%size != 0 */
		if (k > k_scatter) {
			MPI_Scatter(&A[k_scatter*size*m], m, MPI_DOUBLE, &Q_vector[k_scatter*m], m, MPI_DOUBLE, 0, extra_scatter);
		}
		
		if (rank == 0)
			printf("Starting Gram-Schmidt orthogonalization\n...\n");
		
		/* 	Call the gram_schmidt routine, returning the orthogonalized Q_vector:s and
			corresponding R_vector:s */
		gram_schmidt(Q_vector, Q_recv, R_vector, n, m, k, rank, size, rest);
		
		/* Gather the columns of Q and R */
		for (i = 0; i < k_scatter; i++) {
			MPI_Gather(&Q_vector[i*m], m, MPI_DOUBLE, &Q[i*size*m], m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			MPI_Gather(&R_vector[i*n], n, MPI_DOUBLE, &R[i*size*n], n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		}
		
		/* Gather the extra columns of Q and R in the case where n%size != 0 */
		if (k > k_scatter) {
			MPI_Gather(&Q_vector[k_scatter*m], m, MPI_DOUBLE, &Q[k_scatter*size*m], m, MPI_DOUBLE, 0, extra_scatter);
			MPI_Gather(&R_vector[k_scatter*n], n, MPI_DOUBLE, &R[k_scatter*size*n], n, MPI_DOUBLE, 0, extra_scatter);
		}
		
		if (rank == 0) {
			t = MPI_Wtime() - t;
			printf("Wall time: %1.6f\n", t);
		}
	}

	/* 	Check correctness of results
	Note: This part is serial and involves matrix multiplication, which
	means that it is slow for large matrices. */
	
	if (rank == 0) {
		mmul(Q, R, QR_multiply, m, n);
	
		int not_orthogonal = 0;
		for (i = 0; i < n; i++) {
			for (j = i+1; j < n; j++) {
				if (fabs(dot_product(&Q[i*m], &Q[j*m], m)) > 1e-10) {
					printf("Vectors of Q are not orthogonal.\n");
					not_orthogonal = 1;
					break;
				}
			}
			if (not_orthogonal == 1)
				break;
		}
	
		int not_equal = 0;
		for (i = 0; i < n; i++) {
			for (j = 0; j < m; j++) {
				if (fabs(A[i*m+j] - QR_multiply[i*m+j]) > 1e-10) {
					printf("A and QR not equal.\n");
					not_equal = 1;
					break;
				}
			}
			if (not_equal == 1)
				break;
		}

		if (not_orthogonal+not_equal == 0)
			printf("QR-factorization successful!\n");
	}
	
	/* Print QR product */
/* 	if (rank == 0) {
		printf("\nQR:\n");
		for (i = 0; i < m; i++) {
			for (j = 0; j < n; j++) {
				printf("%f  ", QR_multiply[j*m+i]);
			}
			printf("\n");
		}
	} */
	
	/* Print R */
/* 	if (rank == 0) {
		printf("\nR:\n");
		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++) {
				printf("%f ", R[j*n+i]);
			}
			printf("\n");
		}
	} */
	
	/* Free allocated memory */
	if (rank == 0) {
		free(A); free(Q); free(R); free(QR_multiply);
	}
	if (size > 1) {
		free(Q_vector); free(Q_recv); free(R_vector);
	}
	
	MPI_Finalize();
}