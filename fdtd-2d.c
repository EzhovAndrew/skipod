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
   int numworkers,
   int taskid
)
{
	int averow, extra, offset, mtype, rows;
	MPI_Status status;
	if (taskid == MASTER) {
		averow = nx / numworkers;
        extra = nx % numworkers;
        offset = 0;
        mtype = FROM_MASTER;

		for (int dest = 1; dest <= numworkers; dest++) {
			rows = (dest <= extra) ? averow + 1 : averow;
			MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
			MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
			offset = offset + rows;
		}

		mtype = TO_MASTER;
		offset = 0;
		int source;
		for (source = 1; source <= numworkers; source++) {
			rows = (source <= extra) ? averow + 1 : averow;
			MPI_Recv(&ex[offset][0], rows * ny, MPI_DOUBLE, source, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&ey[offset][0], rows * ny, MPI_DOUBLE, source, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&hz[offset][0], rows * ny, MPI_DOUBLE, source, mtype, MPI_COMM_WORLD, &status);
			offset = offset + rows;
		}
	}

	if (taskid > MASTER) {
		mtype = FROM_MASTER;
		MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
		MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
		int i, j;
		for (i = offset; i < offset + rows; i++)
			for (j = 0; j < ny; j++)
			{
				ex[i][j] = ((double) i / nx) * (j+1);
				ey[i][j] = ((double) i / ny) * (j+2);
				hz[i][j] = ((double) i / ny) * (j+3);
			}
		mtype = TO_MASTER;
		MPI_Send(&ex[offset][0], rows * ny, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
		MPI_Send(&ey[offset][0], rows * ny, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
		MPI_Send(&hz[offset][0], rows * ny, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
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
        #pragma omp parallel for default(none) private(j) shared(ny, ey, t)
        for (j = 0; j < ny; j++)
          ey[0][j] = t;

        #pragma omp parallel for default(none) private(i, j) shared(nx, ny, ex, ey, hz)
        for (i = 1; i < nx; i++)
          for (j = 1; j < ny; j++) {
            ex[i][j] -= 0.5*(hz[i][j]-hz[i][j-1]);
            ey[i][j] -= 0.5*(hz[i][j]-hz[i-1][j]);
          }

        #pragma omp parallel for default(none) private(i) shared(nx, ey, hz)
        for (i = 1; i < nx; i++)
            ey[i][0] -= 0.5*(hz[i][0]-hz[i-1][0]);

        #pragma omp parallel for default(none) private(j) shared(ny, ex, hz)
        for (j = 1; j < ny; j++)
            ex[0][j] -= 0.5*(hz[0][j]-hz[0][j-1]);

        #pragma omp parallel for default(none) private(i, j) shared(nx, ny, ex, ey, hz)
        for (i = 0; i < nx - 1; i++)
          for (j = 0; j < ny - 1; j++)
            hz[i][j] -= 0.7* (ex[i][j+1] - ex[i][j] + ey[i+1][j] - ey[i][j]);
    }
}


int main(int argc, char** argv) {
	int numtasks;
	int taskid;
	int numworkers;
	int mtype;

	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
	MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
	numworkers = numtasks - 1;

	int tmax = TMAX;
	int nx = NX;
	int ny = NY;
	double (*ex)[nx][ny];
	ex = (double(*)[nx][ny])malloc ((nx) * (ny) * sizeof(double));

	double (*ey)[nx][ny];
	ey = (double(*)[nx][ny])malloc ((nx) * (ny) * sizeof(double));

	double (*hz)[nx][ny];
	hz = (double(*)[nx][ny])malloc ((nx) * (ny) * sizeof(double));
	init_array (
		tmax, nx, ny,
		*ex,
		*ey,
		*hz,
		numworkers,
		taskid
	);
	// double bench_t_start;
	// if (taskid == MASTER) {
	// 	bench_t_start = MPI_Wtime();
	// }

	// kernel_fdtd_2d (
	// 	tmax, nx, ny,
	// 	*ex,
	// 	*ey,
	// 	*hz
	// );

	// if (taskid == MASTER) {
	// 	double bench_t_end = MPI_Wtime();
	// 	printf ("Time in seconds = %0.6lf\n", bench_t_end - bench_t_start);
	// }

	// free((void*)ex);
	// free((void*)ey);
	// free((void*)hz);
	MPI_Finalize();
	return 0;
}
