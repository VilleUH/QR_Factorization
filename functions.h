#ifndef FUNCTIONS_H
#define FUNCTIONS_H

void mmul(double *A, double *B, double *C, int m, int n);
void normalize(double *vector, double scalar, int m);
void projection(double *vector, double scalar, double *vector_projection, int m);
double dot_product(double *vector_1, double *vector_2, int m);
void normalize_non_contiguous(double *vector, double scalar, int m, int n);
void projection_non_contiguous(double *vector, double scalar, double *vector_projection, int m, int n);
double dot_product_non_contiguous(double *vector_1, double *vector_2, int m, int n);

#endif