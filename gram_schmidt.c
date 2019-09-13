#include <mpi.h>
#include <math.h>
#include "functions.h"

/* The Modified Gram-Schmidt process
Note: One iteration (as referred to in the comments) is defined as the projection
of all vectors onto the space orthogonal to an orthonormal vector q_i */
void gram_schmidt(double *Q_vector, double *Q_recv, double *R_vector, int n, int m, int k, int rank, int size, int catch_last) {
	int i, j, s, rank_recv;
	MPI_Request request_send, request_recv;
	MPI_Status status_send, status_recv;
	
	/* The rank of the process that this process receives from */
	rank_recv = rank-1;
	if (rank == 0)
		rank_recv = size-1;
	
	/* Start up phase */
	
	/* 	Process 0 will only normalize the vector q_1, send it to the rest of the processes
	and project its remaining vectors onto the space orthogonal to q_1, during the
	start up phase. */
	if (rank == 0) {
		
		/* Calculate r_11, normalize q_1 and send it to the rest of the proceses */
		R_vector[rank] = sqrt(dot_product(&Q_vector[0], &Q_vector[0], m));
		normalize(&Q_vector[0], R_vector[rank], m);
		MPI_Isend(&Q_vector[0], m, MPI_DOUBLE, (rank+1)%size, 0, MPI_COMM_WORLD, &request_send);
		
		/* Project the remaining vectors onto the space orthogonal to q_1 */
		for (j = 1; j < k; j++) {
			R_vector[j*n] = dot_product(&Q_vector[0], &Q_vector[j*m], m);
			projection(&Q_vector[j*m], R_vector[j*n], &Q_vector[0], m);
		}
		
		/* Mask the communication with computations by using non-blocking send */
		MPI_Wait(&request_send, &status_send);
	} 
	else 
		
	/* 	The processes will stay in the start up phase for a number of iterations
	corresponding to their rank. */
	{
		for (i = 0; i < rank; i++) {
			/* Receive (into Q_recv) the current iteration's vector that is to be projected onto,
				from process rank_recv. */
			MPI_Irecv(Q_recv, m, MPI_DOUBLE, rank_recv, 0, MPI_COMM_WORLD, &request_recv);
			MPI_Wait(&request_recv, &status_recv);
			
/* 			The last process should not forward q_1, since this process is the last
			process to project its vectors onto q_1 (process 0 already has q_1). */
			if (rank == size-1 && i == 0) {
				// Do nothing
			} else {
				MPI_Isend(Q_recv, m, MPI_DOUBLE, (rank+1)%size, 0, MPI_COMM_WORLD, &request_send);
			}
			
			/* Project the remaining vectors onto the space orthogonal to Q_recv */
			for (j = 0; j < k; j++) {
				R_vector[j*n+i] = dot_product(Q_recv, &Q_vector[j*m], m);
				projection(&Q_vector[j*m], R_vector[j*n+i], Q_recv, m);
			}
			
			if (rank == size-1 && i == 0) {
			// Do nothing
			} else {
				MPI_Wait(&request_send, &status_send);
			}
		}
		
/* 		A process ends the start up phase by normalizing the vector corresponding to
		its rank (which is then a finished orthonormal vector of Q), sends it to the
		rest of the processes and projects its remaining vectors onto the space orthogonal
		to the newly created orthonormal vector. */
		R_vector[rank] = sqrt(dot_product(&Q_vector[0], &Q_vector[0], m));
		normalize(&Q_vector[0], R_vector[rank], m);
		MPI_Isend(&Q_vector[0], m, MPI_DOUBLE, (rank+1)%size, 0, MPI_COMM_WORLD, &request_send);
		for (j = 1; j < k; j++) {
			R_vector[j*n+rank] = dot_product(&Q_vector[0], &Q_vector[j*m], m);
			projection(&Q_vector[j*m], R_vector[j*n+rank], &Q_vector[0], m);
		}
		MPI_Wait(&request_send, &status_send);
	}
	
	/* The regular phase consists of receiving the current iteration's vector to project
	onto, i.e. Q_recv, and (unless this process is the last process to receive this
	particular Q_recv) send it to the next process using non-blocking communication,
	and masking the communication with the projections of the vectors of this process
	onto the space orthogonal to Q_recv. At the end of the outer loop, the process
	normalizes the vector corresponding to the current iteration and sends it to
	the next process, before projecting its remaining vectors onto the space orthogonal
	to the newly created orthonormal vector. */
	
	/* 	Each process will create a total of k orthonormal vectors, the first one created
	in the start up phase. */
	for (s = 1; s < k; s++) {
		
		/* 	The first part of the regular phase, where Q_recv is not forwarded to the next
		process, because this process is the last one that needs Q_recv. Each of the
		vectors of this process is projected onto the space orthogonal to Q_recv. */
		MPI_Irecv(Q_recv, m, MPI_DOUBLE, rank_recv, 0, MPI_COMM_WORLD, &request_recv);
		MPI_Wait(&request_recv, &status_recv);
		for (j = s; j < k; j++) {
			R_vector[j*n+(s-1)*size+rank+1] = dot_product(Q_recv, &Q_vector[j*m], m);
			projection(&Q_vector[j*m], R_vector[j*n+(s-1)*size+rank+1], Q_recv, m);
		}
		
		/* 	The middle part of the regular phase where Q_recv is forwarded to the next
		process, and the vectors of this process is projected onto the space orthogonal
		to Q_recv. */
		for (i = 1; i < size-1; i++) {
			MPI_Irecv(Q_recv, m, MPI_DOUBLE, rank_recv, 0, MPI_COMM_WORLD, &request_recv);
			MPI_Wait(&request_recv, &status_recv);
			MPI_Isend(Q_recv, m, MPI_DOUBLE, (rank+1)%size, 0, MPI_COMM_WORLD, &request_send);
			for (j = s; j < k; j++) {
				R_vector[j*n+(s-1)*size+rank+1+i] = dot_product(Q_recv, &Q_vector[j*m], m);
				projection(&Q_vector[j*m], R_vector[j*n+(s-1)*size+rank+1+i], Q_recv, m);
			}
			MPI_Wait(&request_send, &status_send);
		}
		
		/* 	The last part of the regular phase where this process has a vector that is ready
		for normalization. The process sends this normalized vector to the next process
		and projects its remaining vectors onto the space orthogonal to the newly created
		orthonormal vector. This happens every p (== size) iterations, where p is the number of
		processes. */
		R_vector[s*n+s*size+rank] = sqrt(dot_product(&Q_vector[s*m], &Q_vector[s*m], m));
		normalize(&Q_vector[s*m], R_vector[s*n+s*size+rank], m);
		MPI_Isend(&Q_vector[s*m], m, MPI_DOUBLE, (rank+1)%size, 0, MPI_COMM_WORLD, &request_send);
		for (j = s+1; j < k; j++) {
			R_vector[j*n+s*size+rank] = dot_product(&Q_vector[s*m], &Q_vector[j*m], m);
			projection(&Q_vector[j*m], R_vector[j*n+s*size+rank], &Q_vector[s*m], m);
		}
		MPI_Wait(&request_send, &status_send);
	}
	
	/* 	The last process to exit the regular phase will issue size-1 number of sends that
	is received by the process with rank == catch_last. In the main function
	catch_last == rest, because rest=n%size will be the rank of the process that comes
	after the last process to exit the regular phase. */
	if (rank == catch_last) {
		for (i = 0; i < size-1; i++) {
			MPI_Irecv(Q_recv, m, MPI_DOUBLE, rank_recv, 0, MPI_COMM_WORLD, &request_recv);
			MPI_Wait(&request_recv, &status_recv);
		}
	}
	
}