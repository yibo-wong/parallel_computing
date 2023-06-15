#ifndef SCALAPACK_CONNECTOR_H
#define SCALAPACK_CONNECTOR_H

extern "C" void blacs_get_(int *, int *, int *);
extern "C" void blacs_pinfo_(int *, int *);
extern "C" void blacs_gridinit_(int *, char *, int *, int *);
extern "C" void blacs_gridinfo_(int *, int *, int *, int *, int *);
extern "C" void descinit_(int *, int *, int *, int *, int *, int *, int *,
                          int *, int *, int *);
extern "C" void blacs_gridexit_(int *);
extern "C" int numroc_(int *, int *, int *, int *, int *);
extern "C" void pdsyev_(char *jobz, char *uplo, int *n, double *a, int *ia,
                        int *ja, int *desca, double *w, double *z, int *iz,
                        int *jz, int *descz, double *work, int *lwork,
                        int *info);
extern "C" void pdelset_(double *mat, const int *i, const int *j,
                         const int *desc, const double *a);

#endif
