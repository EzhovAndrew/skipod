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
		averow = (numworkers != 0) ? nx / numworkers : 0;
        extra =  (numworkers != 0) ? nx % numworkers : nx;
        offset = 0;
        mtype = FROM_MASTER;
		if (DEBUG) {
			printf("nx = %d\n", nx);
		}
		for (dest = 1; dest <= numworkers; dest++) {
			MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
			if (DEBUG) {
				printf("master has sent offset - %d\n", offset);
			}
			MPI_Send(&averow, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
			if (DEBUG) {
				printf("master has sent averow - %d\n", averow);
			}
			offset = offset + averow;
		}
		if (DEBUG) {
			printf("master offset = %d\n", offset);
			printf("offset + extra = %d\n", offset + extra);
		}
		for (i = offset; i < offset + extra; i++)
			for (j = 0; j < ny; j++)
			{
				ex[i][j] = ((double) i / nx) * (j+1);
				ey[i][j] = ((double) i / ny) * (j+2);
				hz[i][j] = ((double) i / ny) * (j+3);
			}

		mtype = TO_MASTER;
		offset = 0;
		int source;
		for (source = 1; source <= numworkers; source++) {
			MPI_Recv(&ex[offset][0], averow * ny, MPI_DOUBLE, source, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&ey[offset][0], averow * ny, MPI_DOUBLE, source, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&hz[offset][0], averow * ny, MPI_DOUBLE, source, mtype, MPI_COMM_WORLD, &status);
			offset = offset + averow;
		}
	}

	if (taskid > MASTER) {
		mtype = FROM_MASTER;
		MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
		MPI_Recv(&averow, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
		if (DEBUG) {
			printf("salve recieved offset = %d\n", offset);
			printf("salve recieved averow = %d\n", averow);
		}
		for (i = offset; i < offset + averow; i++)
			for (j = 0; j < ny; j++)
			{
				ex[i][j] = ((double) i / nx) * (j+1);
				ey[i][j] = ((double) i / ny) * (j+2);
				hz[i][j] = ((double) i / ny) * (j+3);
			}
		mtype = TO_MASTER;
		MPI_Send(&ex[offset][0], averow * ny, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
		MPI_Send(&ey[offset][0], averow * ny, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
		MPI_Send(&hz[offset][0], averow * ny, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
	}
}

static
void kernel_fdtd_2d(
	int tmax,
	int nx,
	int ny,
	double ex[ nx][ny],
	double ey[ nx][ny],
	double hz[ nx][ny],
	int numworkers,
	int taskid
)
{
	int averow, extra, offset, mtype, rows, t, i, j, dest, source;
	MPI_Status status;
	for(t = 0; t < tmax; t++)
	{
		if (taskid == MASTER) {
			for (j = 0; j < ny; j++)
				ey[0][j] = t;

			if (DEBUG) {
				printf("iteration!\n");
			}
			averow = (numworkers != 0) ? (nx - 1) / numworkers : 0;
        	extra =  (numworkers != 0) ? (nx - 1) % numworkers : nx;
        	offset = 1;
        	mtype = FROM_MASTER;

			if (DEBUG) {
				printf("averow - %d\n", averow);
				printf("extra - %d\n", extra);
			}

			for (dest = 1; dest <= numworkers; dest++) {
				MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
				MPI_Send(&averow, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
				MPI_Send(&ex[offset][0], averow * ny, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
				MPI_Send(&ey[offset][0], averow * ny, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
				MPI_Send(&hz[offset - 1][0], (averow + 1) * ny, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
				offset = offset + averow;
			}

			for (i = offset; i < offset + extra; i++)
				for (j = 1; j < ny; j++) {
					ex[i][j] -= 0.5*(hz[i][j]-hz[i][j-1]);
					ey[i][j] -= 0.5*(hz[i][j]-hz[i-1][j]);
				}

			offset = 1;
			mtype = TO_MASTER;
			for (source = 1; source <= numworkers; source++) {
				MPI_Recv(&ex[offset][0], averow * ny, MPI_DOUBLE, source, mtype, MPI_COMM_WORLD, &status);
				MPI_Recv(&ey[offset][0], averow * ny, MPI_DOUBLE, source, mtype, MPI_COMM_WORLD, &status);
				offset = offset + averow;
			}

			MPI_Barrier(MPI_COMM_WORLD);

			if (DEBUG) {
				printf("after first barrier\n");
			}

			for (j = 1; j < ny; j++)
				ex[0][j] -= 0.5*(hz[0][j]-hz[0][j-1]);

			offset = 0;
        	mtype = FROM_MASTER;
			for (dest = 1; dest <= numworkers; dest++) {
				MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
				MPI_Send(&averow, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
				MPI_Send(&ex[offset][0], averow * ny, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
				MPI_Send(&ey[offset][0], (averow + 1) * ny, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
				MPI_Send(&hz[offset][0], averow * ny, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
				offset = offset + averow;
			}

			if (DEBUG) {
				printf("after send in second block\n");
			}

			for (i = offset; i < offset + extra - 1; i++)
				for (j = 0; j < ny - 1; j++)
					hz[i][j] -= 0.7* (ex[i][j+1] - ex[i][j] + ey[i+1][j] - ey[i][j]);

			if (DEBUG) {
				printf("before recieve in second block\n");
			}
			offset = 0;
			mtype = TO_MASTER;
			for (source = 1; source <= numworkers; source++) {
				MPI_Recv(&hz[offset][0], averow * ny, MPI_DOUBLE, source, mtype, MPI_COMM_WORLD, &status);
				offset = offset + averow;
			}

			if (DEBUG) {
				printf("end of iteration!\n\n\n");
			}

			MPI_Barrier(MPI_COMM_WORLD);
		}

		if (taskid > MASTER) {
			mtype = FROM_MASTER;
			MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&averow, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&ex[offset][0], averow * ny, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&ey[offset][0], averow * ny, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&hz[offset - 1][0], (averow + 1) * ny, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);

			for (i = offset; i < offset + averow; i++)
				for (j = 1; j < ny; j++) {
					ex[i][j] -= 0.5*(hz[i][j]-hz[i][j-1]);
					ey[i][j] -= 0.5*(hz[i][j]-hz[i-1][j]);
				}

			for (i = offset; i < offset + averow; i++)
				ey[i][0] -= 0.5*(hz[i][0]-hz[i-1][0]);
			
			mtype = TO_MASTER;
			MPI_Send(&ex[offset][0], averow * ny, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
			MPI_Send(&ey[offset][0], averow * ny, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);

			MPI_Barrier(MPI_COMM_WORLD);

			mtype = FROM_MASTER;
			MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&averow, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&ex[offset][0], averow * ny, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&ey[offset][0], (averow + 1) * ny, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&hz[offset][0], averow * ny, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);

			for (i = offset; i < offset + averow; i++)
				for (j = 0; j < ny - 1; j++)
					hz[i][j] -= 0.7 * (ex[i][j+1] - ex[i][j] + ey[i+1][j] - ey[i][j]);

			mtype = TO_MASTER;
			MPI_Send(&hz[offset][0], averow * ny, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);

			MPI_Barrier(MPI_COMM_WORLD);
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

	int tmax = 10;
	int nx = 10;
	int ny = 15;
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
		numtasks - 1,
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
