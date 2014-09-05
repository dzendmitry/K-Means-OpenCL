#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <locale.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define SUCCESS 0
#define FAILURE 1
#define ARGS 4
#define BUFFER 256

float *data = NULL; // �������� ������.
int *point_cluster = NULL; // �������� �����->�������.
int *points_in_cluster = NULL; // ���������� ����� � ��������.
float *prev_centroids = NULL; // ���������� ���������� ����������.
float *centroids = NULL; // ������� ���������� ����������.
float *center_mass = NULL; // ����� ����.

char *OUTPUT = "output.txt"; // ���� � ������������ ������.

// OpenCL ����������.
cl_kernel kernel;
cl_device_id device;
cl_event event;
cl_program program;
cl_mem dataBuffer;
cl_mem prevCentroidsBuffer;
cl_mem centroidsBuffer;
cl_mem pointClusterBuffer;
cl_mem pointsInClBuffer;
cl_mem cMassBuffer;
cl_command_queue commandQueue;
cl_context context;

// ����� �������.
void help() 
{
	printf("�������� ������ ��.��. 220611 ������ �.�.\n\n");
	printf("������������ ���������� ������ k-������� ��� ��������������\n");
	printf("������������� �������� � ������������ �������������� ���������.\n\n");
	printf("�������������: KMeans.exe -n int -c int -f string [-e 0.1]\n");
	printf("�������� ����������:\n");
	printf("-n int - �������� ������������.\n");
	printf("-c int - ���������� ���������.\n");
	printf("-f string - ���� � �������.\n");
	printf("-e float - ����������� �� ��������� 0.1.\n");
	printf("\n");
}

// ������ ���� ����.
char* readKernel(char *file)
{
	FILE *file_desc = fopen(file, "rb");
	if(file_desc == NULL)
	{
		printf("������! �������� ����� %s.\n", file);
		return NULL;
	}
	if(fseek(file_desc, 0, SEEK_END) != 0)
	{
		printf("������! ���� ����. fseek(END).\n");
		return NULL;
	}
	long int size = ftell(file_desc);
	if(size == -1L)
	{
		printf("������! ���� ����. ftell()");
		return NULL;
	}
	if(fseek(file_desc, 0, SEEK_SET) != 0)
	{
		printf("������! ���� ����. fseek(BEGIN).\n");
		return NULL;
	}
	char *source = (char*)calloc((size + 1), sizeof(char));
	if(source == NULL)
	{
		printf("������! ��������� ������ ��� ���������� kmeans_kernel.\n");
		fclose(file_desc);
		return NULL;
	}
	if(fread(source, 1, size, file_desc) != size)
	{
		printf("������! ������ ����� %s.\n", file);
		free(source);
		fclose(file_desc);
		return NULL;
	}
	source[size] = '\0';
	return source;
}

// ���������� ����������� ������.
void writeResult(float *centroids, size_t c_size, float *data, size_t d_size, int n, int *point_cluster)
{
	FILE *file_out = fopen(OUTPUT, "w");
	if(file_out == NULL)
	{
		printf("������! �������� ����� %s.\n", OUTPUT);
		return;
	}
	for(int i = 0; i < c_size; i++)
	{
		fprintf(file_out, "%f\t", centroids[i]);
		if( (i+1) % n == 0 )
			fprintf(file_out, "0\n");
	}
	for(int i = 0; i < d_size; i++)
	{
		fprintf(file_out, "%f\t", data[i]);
		if( (i+1) % n == 0 )
			fprintf(file_out, "%d\n", point_cluster[i / n] + 1);
	}
	fclose(file_out);
}

// ���������� ���������� ������.
void clean()
{
	if(data != NULL)
		free(data);
	if(point_cluster != NULL)
		free(point_cluster);
	if(points_in_cluster != NULL)
		free(points_in_cluster);
	if(prev_centroids != NULL)
		free(prev_centroids);
	if(centroids != NULL)
		free(centroids);
	if(center_mass != NULL)
		free(center_mass);

	if(kernel)
		clReleaseKernel(kernel);
	if(device)
		clReleaseDevice(device);
	if(event)
		clReleaseEvent(event);
	if(program)
		clReleaseProgram(program);
	if(dataBuffer)
		clReleaseMemObject(dataBuffer);
	if(prevCentroidsBuffer)
		clReleaseMemObject(prevCentroidsBuffer);
	if(centroidsBuffer)
		clReleaseMemObject(centroidsBuffer);
	if(pointClusterBuffer)
		clReleaseMemObject(pointClusterBuffer);
	if(pointsInClBuffer)
		clReleaseMemObject(pointsInClBuffer);
	if(cMassBuffer)
		clReleaseMemObject(cMassBuffer);
	if(commandQueue)
		clReleaseCommandQueue(commandQueue);
	if(context)
		clReleaseContext(context);
}

int main(int argc, char *argv[])
{
	setlocale(LC_CTYPE, "Russian");
	if(argc == 1)
	{
		help();
		return FAILURE;
	}

	if( ((argc - 1) % 2) != 0 )
	{
		printf("������! ������� �� ��� �������� ��� ���������� ��� ������� ������.\n");
		help();
		return FAILURE;
	}

	int n = 0; // �������� ������������.
	int clusters = 0; // ���������� ���������.
	float eps = 0.1;
	char *file = NULL; // ���� � �������.
	bool valid_args[ARGS];
	for(int i = 0; i < ARGS; i++)
		valid_args[i] = false;
	valid_args[3] = true; // �������������� �������� -e.
	
	/** ��������� ���������� ���������� ������ **/
	
	for(int i = 0; i < argc; i++)
	{
		if(!strncmp(argv[i], "-n", 2))
		{
			if( (i + 1) >= argc )
			{
				printf("������! �� ������� �������� ��� ��������� -n.\n");
				help();
				return FAILURE;
			}
			n = atoi(argv[i + 1]);
			if(n <= 0)
			{
				printf("������! ���������� ������������� �������� -n � ����� ������������� �����.\n");
				help();
				return FAILURE;
			}
			i++;
			valid_args[0] = true;
		} 
		else if(!strncmp(argv[i], "-c", 2))
		{
			if( (i + 1) >= argc )
			{
				printf("������! �� ������� �������� ��� ��������� -c.\n");
				help();
				return FAILURE;
			}
			clusters = atoi(argv[i + 1]);
			if(clusters <= 0)
			{
				printf("������! ���������� ������������� �������� -c � ����� ������������� �����.\n");
				help();
				return FAILURE;
			}
			i++;
			valid_args[1] = true;
		} 
		else if(!strncmp(argv[i], "-f", 2))
		{
			if( (i + 1) >= argc )
			{
				printf("������! �� ������� �������� ��� ��������� -f.\n");
				help();
				return FAILURE;
			}
			file = argv[i + 1];
			i++;
			valid_args[2] = true;
		} 
		else if(!strncmp(argv[i], "-e", 2))
		{
			if( (i + 1) >= argc )
			{
				printf("������! �� ������� �������� ��� ��������� -e.\n");
				help();
				return FAILURE;
			}
			eps = atof(argv[i + 1]);
			if(eps < 0)
			{
				printf("������! ���������� ������������� �������� -e � ������� ��������������� �����.\n");
				help();
				return FAILURE;
			}
			i++;
			valid_args[3] = true;
		}
	}

	for(int i = 0; i < ARGS; i++)
	{
		if(!valid_args[i])
		{
			printf("������! ������� �� ��� ���������.\n");
			help();
			return FAILURE;
		}
	}

	/** ������ ������� ������ �� �����. **/

	FILE *file_desc = fopen(file, "r");
	if(file_desc == NULL)
	{
		printf("������! ���������� ������� ���� � �������.\n");
		return FAILURE;
	}
	unsigned data_length = BUFFER; // ������� ������ ������� data.
	data = (float*)calloc(data_length, sizeof(float)); // �������� ������.
	if(data == NULL)
	{
		printf("������! ���������� �������� ������.");
		return FAILURE;
	}
	unsigned real_data_length = 0;
	unsigned p = 0;
	while(!feof(file_desc))
	{
		if(fscanf(file_desc, "%f", &data[real_data_length]) != 1)
		{
			printf("������! �� ������� ������� ���� �� �������� � ����� ������. %d\n", p);
			clean();
			return FAILURE;
		}
		if(real_data_length == (data_length - 1))
		{
			data_length += BUFFER;
			float *old_datap = data;
			data = (float*)realloc(data, data_length * sizeof(float));
			if(data == NULL)
			{
				printf("������! ������� ����� ������. ������������ ����������� ������.\n");
				free(old_datap);
				return FAILURE;
			}
		}
		real_data_length++;
		p++;
	}
	fclose(file_desc);

	printf("��������� %d �����.\n", real_data_length);

	if((real_data_length % n) != 0)
	{
		printf("������! ������ �����������. ���������� �������� ����� ���������� ����� � %d ������ ������������.", n);
		clean();
		return FAILURE;
	}
	int points = real_data_length / n; // ���������� ����� � ������.

	/** ��������� ������. **/
	point_cluster = (int*)calloc(points, sizeof(int));
	if(point_cluster == NULL)
	{
		printf("������! ���������� �������� ������ ��� ������ �����->�������.");
		clean();
		return FAILURE;
	}
	points_in_cluster = (int*)calloc(clusters, sizeof(int));
	if(points_in_cluster == NULL)
	{
		printf("������! ���������� �������� ������ ��� ������ ���������� ����� � ��������.");
		clean();
		return FAILURE;
	}
	prev_centroids = (float*)calloc(clusters * n, sizeof(float));
	if(prev_centroids == NULL)
	{
		printf("������! ���������� �������� ������ ��� ������ ��� �������� ���������� ��������� ����������.");
		clean();
		return FAILURE;
	}
	centroids = (float*)calloc(clusters * n, sizeof(float));
	if(prev_centroids == NULL)
	{
		printf("������! ���������� �������� ������ ��� ������ ��� �������� ������� ��������� ����������.");
		clean();
		return FAILURE;
	}
	center_mass = (float*)calloc(clusters * n, sizeof(float));
	if(prev_centroids == NULL)
	{
		printf("������! ���������� �������� ������ ��� ������ ������ ����.");
		clean();
		return FAILURE;
	}
	
	/** ���������� ��������� ��������� ����������. **/
	int step = points / clusters;
	for(int i = 0, j = 0; i < clusters; i++, j += step)
		for(int k = (i * n), m = 0; k < ((i+1) * n); k++, m++)
			centroids[k] = data[j * n + m];

	/** ������ OpenCL **/

	/* ��� 1: ����� ���������. */
	cl_uint num_platforms;	// ����� ���������.
	cl_platform_id platform = NULL;
	cl_int status = clGetPlatformIDs(0, NULL, &num_platforms);
	if (status != CL_SUCCESS)
	{
		printf("������! ��������� ���������� ��������.\n");
		clean();
		return FAILURE;
	}

	/* ����� ������ ���������. */
	if(num_platforms > 0)
	{
		cl_platform_id* platforms = (cl_platform_id*)calloc(num_platforms, sizeof(cl_platform_id));
		if(platforms == NULL)
		{
			printf("������! ��������� ������ ��� ������ ��������.\n");
			clean();
			return FAILURE;
		}
		status = clGetPlatformIDs(num_platforms, platforms, NULL);
		if(status != CL_SUCCESS)
		{
			printf("������! ��������� ������� ��������.\n");
			free(platforms);
			clean();
			return FAILURE;
		}
		platform = platforms[0];
		free(platforms);
	}

	/* ��� 2: ������ � ����� GPU ��� CPU ����������. */
	cl_uint				num_devices = 0;
	cl_device_id        *devices;
	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);	
	if(status != CL_SUCCESS)
	{
		printf("������! ��������� ������� GPU ���������.\n");
		clean();
		return FAILURE;
	}
	if (num_devices == 0) // GPU �� �������.
	{
		printf("GPU �� �������.\n");
		printf("��������! ������������ CPU.\n");
		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, &num_devices);	
		if(status != CL_SUCCESS)
		{
			printf("������! ��������� ������� CPU ���������.\n");
			clean();
			return FAILURE;
		}
		devices = (cl_device_id*)calloc(num_devices, sizeof(cl_device_id));
		if(devices == NULL)
		{
			printf("������! ��������� ������ ��� ������ ���������.\n");
			clean();
			return FAILURE;
		}
		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, num_devices, devices, NULL);
		if(status != CL_SUCCESS)
		{
			printf("������! ��������� ������ CPU ���������.\n");
			free(devices);
			clean();
			return FAILURE;
		}
	}
	else
	{
		devices = (cl_device_id*)calloc(num_devices, sizeof(cl_device_id));
		if(devices == NULL)
		{
			printf("������! ��������� ������ ��� ������ ���������.\n");
			clean();
			return FAILURE;
		}
		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);
		if(status != CL_SUCCESS)
		{
			printf("������! ��������� ������ GPU ���������.\n");
			free(devices);
			clean();
			return FAILURE;
		}
	}
	device = devices[0];
	free(devices);


	/* ��� 2.1: ����� ���������� �� ���������� */
	char info_buffer[BUFFER];
	cl_bool info_bool;
	cl_uint info_uint;
	size_t info_size_t;
	printf("\n");
	clGetDeviceInfo(device, CL_DEVICE_NAME, BUFFER * sizeof(char), info_buffer, NULL);
	printf("�������� ����������: %s\n", info_buffer);
	clGetDeviceInfo(device, CL_DEVICE_VENDOR, BUFFER * sizeof(char), info_buffer, NULL);
	printf("�������������: %s\n", info_buffer);
	clGetDeviceInfo(device, CL_DEVICE_VERSION, BUFFER * sizeof(char), info_buffer, NULL);
	printf("�������������� ������ OpenCL: %s\n", info_buffer);
	clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, BUFFER * sizeof(char), info_buffer, NULL);
	printf("�������������� ������ ����� C OpenCL: %s\n", info_buffer);
	clGetDeviceInfo(device, CL_DEVICE_COMPILER_AVAILABLE, sizeof(cl_bool), &info_bool, NULL);
	printf("������� �����������: %s\n", info_bool ? "YES" : "NO");
	clGetDeviceInfo(device, CL_DEVICE_LINKER_AVAILABLE, sizeof(cl_bool), &info_bool, NULL);
	printf("������� ����������: %s\n", info_bool ? "YES" : "NO");
	clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &info_uint, NULL);
	printf("����. ������������ ������� �����: %u\n", info_uint);
	clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &info_uint, NULL);
	printf("����. �������� ���������: %u\n", info_uint);
	clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &info_size_t, NULL);
	printf("����. �������� ��������� � ������: %u\n", info_size_t);
	printf("\n");
	const size_t local_work_size = info_size_t; // ���������� work-intem � ��������� ������.
	const size_t global_work_size = local_work_size * ceil((double)points / local_work_size); // ����� work-intems.

	/* ��� 3: �������� ���������.*/
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
	if(status != CL_SUCCESS)
	{
		printf("������! �������� ���������.\n");
		clean();
		return FAILURE;
	}

	/* ��� 4: �������� ������� ������, ��������������� � ����������.*/
	commandQueue = clCreateCommandQueue(context, device, 0, &status);
	if(status != CL_SUCCESS)
	{
		printf("������! �������� ������� ������.\n");
		clean();
		return FAILURE;
	}

	/* ��� 5: �������� ������������ �������. */
	char *filename = "kmeans_kernel.c";
	char *source = readKernel(filename);
	if(source == NULL)
	{
		clean();
		return FAILURE;
	}
	size_t sourceSize[] = {strlen(source)};
	const char *source_const = source;
	program = clCreateProgramWithSource(context, 1, &source_const, sourceSize, &status);
	if(status != CL_SUCCESS)
	{
		printf("������! �������� ������������ �������.\n");
		clean();
		free(source);
		return FAILURE;
	}
	free(source);
	
	/* ��� 6: ������ ���������. */
	status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	if(status != CL_SUCCESS)
	{
		printf("������! ������ ���������.\n");
		clean();
		return FAILURE;
	}

	/* ��� 7: ������������� �������, �������� ������� ��� ����� � ����. */
	dataBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, (real_data_length) * sizeof(float), (void*) data, &status);
	if(status != CL_SUCCESS)
	{
		printf("������! ������������ ����������� ��� �������� ������ ������.\n");
		clean();
		return FAILURE;
	}
	prevCentroidsBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR, clusters * n * sizeof(float), (void*) prev_centroids, &status);
	if(status != CL_SUCCESS)
	{
		printf("������! ������������ ����������� ��� �������� ������ ���������� ����������.\n");
		clean();
		return FAILURE;
	}
	centroidsBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR, clusters * n * sizeof(float), (void*) centroids, &status);
	if(status != CL_SUCCESS)
	{
		printf("������! ������������ ����������� ��� �������� ������ ����������.\n");
		clean();
		return FAILURE;
	}
	pointClusterBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, points * sizeof(int), (void*) point_cluster, &status);
	if(status != CL_SUCCESS)
	{
		printf("������! ������������ ����������� ��� �������� ������ �����->�������.\n");
		clean();
		return FAILURE;
	}
	pointsInClBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR, clusters * sizeof(int), (void*) points_in_cluster, &status);
	if(status != CL_SUCCESS)
	{
		printf("������! ������������ ����������� ��� �������� ������ ���������� ����� � ��������.\n");
		clean();
		return FAILURE;
	}
	cMassBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR, clusters * n * sizeof(float), (void*) center_mass, &status);
	if(status != CL_SUCCESS)
	{
		printf("������! ������������ ����������� ��� �������� ������ ������ ����.\n");
		clean();
		return FAILURE;
	}

	/* ��� 8: �������� ����. */
	kernel = clCreateKernel(program, "kmeans", &status);
	if(status != CL_SUCCESS)
	{
		printf("������! �������� ����.\n");
		clean();
		return FAILURE;
	}

	/* ��� 9: ��������� ���������� ����.*/
	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&dataBuffer);
	status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&prevCentroidsBuffer);
	status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&centroidsBuffer);
	status |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&pointClusterBuffer);
	status |= clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&pointsInClBuffer);
	status |= clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&cMassBuffer);
	status |= clSetKernelArg(kernel, 6, sizeof(cl_int), (void *)&n);
	status |= clSetKernelArg(kernel, 7, sizeof(cl_int), (void *)&clusters);
	status |= clSetKernelArg(kernel, 8, sizeof(cl_int), (void *)&points);
	if(status != CL_SUCCESS)
	{
		printf("������! ��������� ���������� ����.\n");
		clean();
		return FAILURE;
	}

	/* ��� 10: ������ ����.*/
	clock_t alg_time = clock();
	while(true)
	{
		for(int i = 0; i < points; i++)
			point_cluster[i] = 0;
		for(int i = 0; i < (clusters * n); i += 1)
			center_mass[i] = 0;
		for(int i = 0; i < clusters; i += 1)
			points_in_cluster[i] = 0;
		
		status = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &event);
		if(status != CL_SUCCESS)
		{
			printf("������! ������ ����.\n");
			clean();
			return FAILURE;
		}
		clock_t start_time = clock();
		status = clWaitForEvents(1, &event);
		if(status != CL_SUCCESS)
		{
			printf("������! �������� ��������� ������ ����.\n");
			clean();
			return FAILURE;
		}
		//printf("������ ���� ������� ���������.\n");
		//printf("����� ������ ����: %f ���.\n", ((float)(clock() - start_time))/CLOCKS_PER_SEC);

		// ��������� ���������� ���������.
		for(int i = 0; i < (clusters * n); i++)
			prev_centroids[i] = centroids[i];

		// ��������� ����� ���������� ����������.
		for(int i = 0; i < clusters; i++)
			for(int j = i * n, k = 0; j < (i * n + n); j++, k++)
				centroids[j] = center_mass[j] / points_in_cluster[i];

		// �������� ���������� ������� ������� ���� � �����������.
		int chk = 0;
		for(int i = 0; i < (n * clusters); i++)
			if(fabs(prev_centroids[i] - centroids[i]) <= eps)
				chk += 1;
		if(chk == (n * clusters))
			break;
	}

	printf("����� ������ ���������: %f ���.\n", ((float)(clock() - alg_time))/CLOCKS_PER_SEC);

	status = clEnqueueReadBuffer(commandQueue, pointClusterBuffer, CL_TRUE, 0, points * sizeof(int), (void*) point_cluster, 0, NULL, NULL);
	if(status != CL_SUCCESS)
	{
		printf("������! ��������� ������������� ����������� ������ �� �����������.\n");
		clean();
		return FAILURE;
	}
	writeResult(centroids, n * clusters, data, real_data_length, n, point_cluster);

	clean();
	return SUCCESS;
}