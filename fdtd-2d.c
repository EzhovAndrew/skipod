/* Include benchmark-specific header. */
#include "fdtd-2d.h"

static
double rtclock()
{
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, NULL);
    if (stat != 0)
      printf ("Error return from gettimeofday: %d", stat);
    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

static
void init_array (int tmax,
   int nx,
   int ny,
   double ex[ nx][ny],
   double ey[ nx][ny],
   double hz[ nx][ny],
   int nThreads)
{
  int i, j;

  #pragma parallel for default(none) private(i, j) shared(ex, ey, hz, nx, ny)
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++)
      {
        ex[i][j] = ((double) i / nx) * (j+1);
        ey[i][j] = ((double) i / ny) * (j+2);
        hz[i][j] = ((double) i / ny) * (j+3);
      }
}

static
void kernel_fdtd_2d(int tmax,
      int nx,
      int ny,
      double ex[ nx][ny],
      double ey[ nx][ny],
      double hz[ nx][ny]
)
{
  int t, i, j;

  for(t = 0; t < tmax; t++)
    {
      #pragma omp parallel for default(none) private(j) shared(ey, ny, t)
      for (j = 0; j < ny; j++)
          ey[0][j] = t;

      #pragma omp parallel for default(none) private(i, j) shared(ey, hz, nx, ny)
      for (i = 1; i < nx; i++)
        for (j = 0; j < ny; j++)
          ey[i][j] -= 0.5*(hz[i][j]-hz[i-1][j]);

      #pragma omp parallel for default(none) private(i, j) shared(ex, hz, nx, ny)
      for (i = 0; i < nx; i++)
        for (j = 1; j < ny; j++)
          ex[i][j] -= 0.5*(hz[i][j]-hz[i][j-1]);

      #pragma omp parallel for default(none) private(i, j) shared(hz, ex, ey, nx, ny)
      for (i = 0; i < nx - 1; i++)
        for (j = 0; j < ny - 1; j++)
          hz[i][j] -= 0.7* (ex[i][j+1] - ex[i][j] + ey[i+1][j] - ey[i][j]);
    }
}


int main(int argc, char** argv) {
  int tmax = TMAX;
  int nx = NX;
  int ny = NY;
  double (*ex)[nx][ny];
  ex = (double(*)[nx][ny])malloc ((nx) * (ny) * sizeof(double));

  double (*ey)[nx][ny];
  ey = (double(*)[nx][ny])malloc ((nx) * (ny) * sizeof(double));

  double (*hz)[nx][ny];
  hz = (double(*)[nx][ny])malloc ((nx) * (ny) * sizeof(double));

  int nThreads = 4;
  // scanf("%d", &nThreads);

  init_array (
    tmax, nx, ny,
    *ex,
    *ey,
    *hz,
    nThreads
  );

  double bench_t_start = rtclock();
  kernel_fdtd_2d (
    tmax, nx, ny,
    *ex,
    *ey,
    *hz
  );

  double bench_t_end = rtclock();
  printf ("Time in seconds = %0.6lf\n", bench_t_end - bench_t_start);
  free((void*)ex);
  free((void*)ey);
  free((void*)hz);

  return 0;
}
