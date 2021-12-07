/* Include benchmark-specific header. */
#include "fdtd-2d.h"


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
	int averow, extra, offset, mtype, rows, i, j, dest;
	MPI_Status status;
	if (taskid == MASTER) {
		averow = nx / numworkers;
        extra =  nx % numworkers;
        offset = 0;
        mtype = FROM_MASTER;
		for (dest = 1; dest < numworkers; dest++) {
			rows = (dest <= extra) ? averow + 1 : averow;
			MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
			MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
			offset = offset + rows;
		}
		for (i = offset; i < offset + averow; i++)
			for (j = 0; j < ny; j++)
			{
				ex[i][j] = ((double) i / nx) * (j+1);
				ey[i][j] = ((double) i / ny) * (j+2);
				hz[i][j] = ((double) i / ny) * (j+3);
			}

		mtype = TO_MASTER;
		offset = 0;
		int source;
		for (source = 1; source < numworkers; source++) {
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
		for (i = 0; i < rows; i++)
			for (j = 0; j < ny; j++)
			{
				ex[i][j] = ((double) (i + offset) / nx) * (j+1);
				ey[i][j] = ((double) (i + offset) / ny) * (j+2);
				hz[i][j] = ((double) (i + offset) / ny) * (j+3);
			}
		mtype = TO_MASTER;
		MPI_Send(&ex[0][0], rows * ny, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
		MPI_Send(&ey[0][0], rows * ny, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
		MPI_Send(&hz[0][0], rows * ny, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
	}
}

static
void kernel_fdtd_2d(
	int tmax,
	int nx,
	int ny,
	double ex[nx][ny],
	double ey[nx][ny],
	double hz[nx][ny],
	int numworkers,
	int taskid
)
{
	int averow, extra, offset, mtype, rows, t, i, j, dest, source;
	MPI_Status status;
	for(t = 0; t < tmax; t++)
	{
		if (taskid == MASTER) {
			if (DEBUG) {
				printf("Iteration!\n");
			}

			for (j = 0; j < ny; j++)
				ey[0][j] = t;

			averow = (nx - 1) / numworkers;
        	extra =  (nx - 1) % numworkers;
        	offset = 1;
        	mtype = FROM_MASTER;
			for (dest = 1; dest <= numworkers; dest++) {
				rows = (dest <= extra) ? averow + 1 : averow;
				MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
				MPI_Send(&ex[offset][0], rows * ny, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
				MPI_Send(&ey[offset][0], rows * ny, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
				MPI_Send(&hz[offset - 1][0], (rows + 1) * ny, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
				offset = offset + rows;
			}

			offset = 1;
			mtype = TO_MASTER;
			for (source = 1; source <= numworkers; source++) {
				rows = (source <= extra) ? averow + 1 : averow;
				MPI_Recv(&ex[offset][0], rows * ny, MPI_DOUBLE, source, mtype, MPI_COMM_WORLD, &status);
				MPI_Recv(&ey[offset][0], rows * ny, MPI_DOUBLE, source, mtype, MPI_COMM_WORLD, &status);
				offset = offset + rows;
			}

			if (DEBUG) {
				printf("after first barrier\n");
			}

			for (j = 1; j < ny; j++)
				ex[0][j] -= 0.5*(hz[0][j]-hz[0][j-1]);

			offset = 0;
        	mtype = FROM_MASTER;
			for (dest = 1; dest <= numworkers; dest++) {
				rows = (dest <= extra) ? averow + 1 : averow;
				MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
				MPI_Send(&ex[offset][0], rows * ny, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
				MPI_Send(&ey[offset][0], (rows + 1) * ny, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
				MPI_Send(&hz[offset][0], rows * ny, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
				offset = offset + rows;
			}

			if (DEBUG) {
				printf("after send in second block\n");
			}

			offset = 0;
			mtype = TO_MASTER;
			for (source = 1; source <= numworkers; source++) {
				rows = (source <= extra) ? averow + 1 : averow;
				MPI_Recv(&hz[offset][0], rows * ny, MPI_DOUBLE, source, mtype, MPI_COMM_WORLD, &status);
				offset = offset + rows;
			}

			if (DEBUG) {
				printf("end of iteration!\n\n\n");
			}
		}

		if (taskid > MASTER) {
			mtype = FROM_MASTER;
			MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&ex[1][0], rows * ny, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&ey[1][0], rows * ny, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&hz[0][0], (rows + 1) * ny, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);

			for (i = 1; i < rows + 1; i++)
				for (j = 1; j < ny; j++) {
					ex[i][j] -= 0.5*(hz[i][j]-hz[i][j-1]);
					ey[i][j] -= 0.5*(hz[i][j]-hz[i-1][j]);
				}

			for (i = 1; i < rows + 1; i++)
				ey[i][0] -= 0.5*(hz[i][0]-hz[i-1][0]);
			
			mtype = TO_MASTER;
			MPI_Send(&ex[1][0], rows * ny, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
			MPI_Send(&ey[1][0], rows * ny, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);

			mtype = FROM_MASTER;
			MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&ex[0][0], rows * ny, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&ey[0][0], (rows + 1) * ny, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&hz[0][0], rows * ny, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);

			for (i = 0; i < rows; i++)
				for (j = 0; j < ny - 1; j++)
					hz[i][j] -= 0.7 * (ex[i][j+1] - ex[i][j] + ey[i+1][j] - ey[i][j]);

			mtype = TO_MASTER;
			MPI_Send(&hz[0][0], rows * ny, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
		}
	}
}


int main(int argc, char** argv) {
	int numtasks;
	int taskid;
	int i, j;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	MPI_Comm_rank(MPI_COMM_WORLD, &taskid);

	int tmax = TMAX;
	int nx = NX;
	int ny = NY;
	int maxrows;
	if (taskid == MASTER) {
		maxrows = nx;
	}
	else {
		maxrows = (nx - 1) / (numtasks - 1) + 2;
	}
	double (*ex)[maxrows][ny];
	ex = (double(*)[maxrows][ny])malloc ((maxrows) * (ny) * sizeof(double));

	double (*ey)[maxrows][ny];
	ey = (double(*)[maxrows][ny])malloc ((maxrows) * (ny) * sizeof(double));

	double (*hz)[maxrows][ny];
	hz = (double(*)[maxrows][ny])malloc ((maxrows) * (ny) * sizeof(double));
	init_array (
		tmax, nx, ny,
		*ex,
		*ey,
		*hz,
		numtasks,
		taskid
	);
	MPI_Barrier(MPI_COMM_WORLD);
	double bench_t_start;
	if (taskid == MASTER) {
		bench_t_start = MPI_Wtime();
	}

	kernel_fdtd_2d (
		tmax, nx, ny,
		*ex,
		*ey,
		*hz,
	 	numtasks - 1,
	 	taskid
	);

	if (taskid == MASTER) {
		double bench_t_end = MPI_Wtime();
		printf ("Time in seconds = %0.6lf\n", bench_t_end - bench_t_start);
	}

	free((void*)ex);
	free((void*)ey);
	free((void*)hz);
	MPI_Finalize();
	return 0;
}
