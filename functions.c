#include <math.h>

/* Matrix multiplication */
void mmul(double *A, double *B, double *C, int m, int n) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			for (int K = 0; K < n; K++) {
				C[j*m+i] = C[j*m+i] + A[K*m+i]*B[K+j*n];
			}
		}
	}
}

/* Vector normalization */
void normalize(double *vector, double scalar, int m) {
	int i;
	for (i = 0; i < m; i++) {
		vector[i] /= scalar;
	}
}

/* Projection of 'vector' onto space orthogonal to 'vector_projection' */
void projection(double *vector, double scalar, double *vector_projection, int m) {
	int i;
	for (i = 0; i < m; i++) {
		vector[i] = vector[i] - scalar*vector_projection[i];
	}
}

/* Inner products of vectors */
double dot_product(double *vector_1, double *vector_2, int m) {
	int i;
	double sum = 0;
	for (i = 0; i < m; i++) {
		sum += vector_1[i]*vector_2[i];
	}
	return sum;
}