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

float *data = NULL; // Исходные данные.
int *point_cluster = NULL; // Привязка точка->кластер.
int *points_in_cluster = NULL; // Количество точек в кластере.
float *prev_centroids = NULL; // Предудущие координаты центроидов.
float *centroids = NULL; // Текущие координаты центроидов.
float *center_mass = NULL; // Центр масс.

char *OUTPUT = "output.txt"; // Файл с результатами работы.

// OpenCL переменные.
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

// Вывод справки.
void help() 
{
	printf("Курсовая работа ст.гр. 220611 Зенина Д.Г.\n\n");
	printf("Параллельная реализация метода k-средних для автоматической\n");
	printf("классификации объектов в пространстве действительных признаков.\n\n");
	printf("Использование: KMeans.exe -n int -c int -f string [-e 0.1]\n");
	printf("Описание параметров:\n");
	printf("-n int - мерность пространства.\n");
	printf("-c int - количество кластеров.\n");
	printf("-f string - файл с данными.\n");
	printf("-e float - погрешность По умолчанию 0.1.\n");
	printf("\n");
}

// Чтение кода ядра.
char* readKernel(char *file)
{
	FILE *file_desc = fopen(file, "rb");
	if(file_desc == NULL)
	{
		printf("Ошибка! Открытие файла %s.\n", file);
		return NULL;
	}
	if(fseek(file_desc, 0, SEEK_END) != 0)
	{
		printf("Ошибка! Файл ядра. fseek(END).\n");
		return NULL;
	}
	long int size = ftell(file_desc);
	if(size == -1L)
	{
		printf("Ошибка! Файл ядра. ftell()");
		return NULL;
	}
	if(fseek(file_desc, 0, SEEK_SET) != 0)
	{
		printf("Ошибка! Файл ядра. fseek(BEGIN).\n");
		return NULL;
	}
	char *source = (char*)calloc((size + 1), sizeof(char));
	if(source == NULL)
	{
		printf("Ошибка! Выделение памяти под содержимое kmeans_kernel.\n");
		fclose(file_desc);
		return NULL;
	}
	if(fread(source, 1, size, file_desc) != size)
	{
		printf("Ошибка! Чтение файла %s.\n", file);
		free(source);
		fclose(file_desc);
		return NULL;
	}
	source[size] = '\0';
	return source;
}

// Сохранение результатов работы.
void writeResult(float *centroids, size_t c_size, float *data, size_t d_size, int n, int *point_cluster)
{
	FILE *file_out = fopen(OUTPUT, "w");
	if(file_out == NULL)
	{
		printf("Ошибка! Открытие файла %s.\n", OUTPUT);
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

// Корректное завершение работы.
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
		printf("Ошибка! Указаны не все значения для параметров или указаны лишние.\n");
		help();
		return FAILURE;
	}

	int n = 0; // Мерность пространства.
	int clusters = 0; // Количество кластеров.
	float eps = 0.1;
	char *file = NULL; // Файл с данными.
	bool valid_args[ARGS];
	for(int i = 0; i < ARGS; i++)
		valid_args[i] = false;
	valid_args[3] = true; // Необязательный параметр -e.
	
	/** Обработка аргументов коммандной строки **/
	
	for(int i = 0; i < argc; i++)
	{
		if(!strncmp(argv[i], "-n", 2))
		{
			if( (i + 1) >= argc )
			{
				printf("Ошибка! Не указано значение для параметра -n.\n");
				help();
				return FAILURE;
			}
			n = atoi(argv[i + 1]);
			if(n <= 0)
			{
				printf("Ошибка! Невозможно преобразовать аргумент -n в целое положительное число.\n");
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
				printf("Ошибка! Не указано значение для параметра -c.\n");
				help();
				return FAILURE;
			}
			clusters = atoi(argv[i + 1]);
			if(clusters <= 0)
			{
				printf("Ошибка! Невозможно преобразовать аргумент -c в целое положительное число.\n");
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
				printf("Ошибка! Не указано значение для параметра -f.\n");
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
				printf("Ошибка! Не указано значение для параметра -e.\n");
				help();
				return FAILURE;
			}
			eps = atof(argv[i + 1]);
			if(eps < 0)
			{
				printf("Ошибка! Невозможно преобразовать аргумент -e в дробное неотрицательное число.\n");
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
			printf("Ошибка! Указаны не все аргументы.\n");
			help();
			return FAILURE;
		}
	}

	/** Чтение входных данных из файла. **/

	FILE *file_desc = fopen(file, "r");
	if(file_desc == NULL)
	{
		printf("Ошибка! Невозможно открыть файл с данными.\n");
		return FAILURE;
	}
	unsigned data_length = BUFFER; // Текущий размер массива data.
	data = (float*)calloc(data_length, sizeof(float)); // Исходные данные.
	if(data == NULL)
	{
		printf("Ошибка! Невозможно выделить память.");
		return FAILURE;
	}
	unsigned real_data_length = 0;
	unsigned p = 0;
	while(!feof(file_desc))
	{
		if(fscanf(file_desc, "%f", &data[real_data_length]) != 1)
		{
			printf("Ошибка! Не удалось считать одно из значений в файле данных. %d\n", p);
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
				printf("Ошибка! Слишком много данных. Недостаточно оперативной памяти.\n");
				free(old_datap);
				return FAILURE;
			}
		}
		real_data_length++;
		p++;
	}
	fclose(file_desc);

	printf("Прочитано %d чисел.\n", real_data_length);

	if((real_data_length % n) != 0)
	{
		printf("Ошибка! Данные некорректны. Невозможно получить целое количество точек в %d мерном пространстве.", n);
		clean();
		return FAILURE;
	}
	int points = real_data_length / n; // Количество точек в данных.

	/** Выделение памяти. **/
	point_cluster = (int*)calloc(points, sizeof(int));
	if(point_cluster == NULL)
	{
		printf("Ошибка! Невозможно выделить память под массив точка->кластер.");
		clean();
		return FAILURE;
	}
	points_in_cluster = (int*)calloc(clusters, sizeof(int));
	if(points_in_cluster == NULL)
	{
		printf("Ошибка! Невозможно выделить память под массив количество точек в кластере.");
		clean();
		return FAILURE;
	}
	prev_centroids = (float*)calloc(clusters * n, sizeof(float));
	if(prev_centroids == NULL)
	{
		printf("Ошибка! Невозможно выделить память под массив для хранения предыдущих координат центроидов.");
		clean();
		return FAILURE;
	}
	centroids = (float*)calloc(clusters * n, sizeof(float));
	if(prev_centroids == NULL)
	{
		printf("Ошибка! Невозможно выделить память под массив для хранения текущих координат центроидов.");
		clean();
		return FAILURE;
	}
	center_mass = (float*)calloc(clusters * n, sizeof(float));
	if(prev_centroids == NULL)
	{
		printf("Ошибка! Невозможно выделить память под массив центра масс.");
		clean();
		return FAILURE;
	}
	
	/** Вычисление начальных координат центроидов. **/
	int step = points / clusters;
	for(int i = 0, j = 0; i < clusters; i++, j += step)
		for(int k = (i * n), m = 0; k < ((i+1) * n); k++, m++)
			centroids[k] = data[j * n + m];

	/** Секция OpenCL **/

	/* Шаг 1: Выбор платформы. */
	cl_uint num_platforms;	// Номер платформы.
	cl_platform_id platform = NULL;
	cl_int status = clGetPlatformIDs(0, NULL, &num_platforms);
	if (status != CL_SUCCESS)
	{
		printf("Ошибка! Получение количества платформ.\n");
		clean();
		return FAILURE;
	}

	/* Выбор первой платформы. */
	if(num_platforms > 0)
	{
		cl_platform_id* platforms = (cl_platform_id*)calloc(num_platforms, sizeof(cl_platform_id));
		if(platforms == NULL)
		{
			printf("Ошибка! Выделение памяти под массив платформ.\n");
			clean();
			return FAILURE;
		}
		status = clGetPlatformIDs(num_platforms, platforms, NULL);
		if(status != CL_SUCCESS)
		{
			printf("Ошибка! Получение массива платформ.\n");
			free(platforms);
			clean();
			return FAILURE;
		}
		platform = platforms[0];
		free(platforms);
	}

	/* Шаг 2: Запрос и выбор GPU или CPU устройства. */
	cl_uint				num_devices = 0;
	cl_device_id        *devices;
	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);	
	if(status != CL_SUCCESS)
	{
		printf("Ошибка! Получение номеров GPU устройств.\n");
		clean();
		return FAILURE;
	}
	if (num_devices == 0) // GPU не найдены.
	{
		printf("GPU не найдены.\n");
		printf("Внимание! Используется CPU.\n");
		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, &num_devices);	
		if(status != CL_SUCCESS)
		{
			printf("Ошибка! Получение номеров CPU устройств.\n");
			clean();
			return FAILURE;
		}
		devices = (cl_device_id*)calloc(num_devices, sizeof(cl_device_id));
		if(devices == NULL)
		{
			printf("Ошибка! Выделение памяти под массив устройств.\n");
			clean();
			return FAILURE;
		}
		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, num_devices, devices, NULL);
		if(status != CL_SUCCESS)
		{
			printf("Ошибка! Получение списка CPU устройств.\n");
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
			printf("Ошибка! Выделение памяти под массив устройств.\n");
			clean();
			return FAILURE;
		}
		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);
		if(status != CL_SUCCESS)
		{
			printf("Ошибка! Получение списка GPU устройств.\n");
			free(devices);
			clean();
			return FAILURE;
		}
	}
	device = devices[0];
	free(devices);


	/* Шаг 2.1: Вывод информации об устройстве */
	char info_buffer[BUFFER];
	cl_bool info_bool;
	cl_uint info_uint;
	size_t info_size_t;
	printf("\n");
	clGetDeviceInfo(device, CL_DEVICE_NAME, BUFFER * sizeof(char), info_buffer, NULL);
	printf("Название устройства: %s\n", info_buffer);
	clGetDeviceInfo(device, CL_DEVICE_VENDOR, BUFFER * sizeof(char), info_buffer, NULL);
	printf("Производитель: %s\n", info_buffer);
	clGetDeviceInfo(device, CL_DEVICE_VERSION, BUFFER * sizeof(char), info_buffer, NULL);
	printf("Поддерживаемая версия OpenCL: %s\n", info_buffer);
	clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, BUFFER * sizeof(char), info_buffer, NULL);
	printf("Поддерживаемая версия языка C OpenCL: %s\n", info_buffer);
	clGetDeviceInfo(device, CL_DEVICE_COMPILER_AVAILABLE, sizeof(cl_bool), &info_bool, NULL);
	printf("Наличие компилятора: %s\n", info_bool ? "YES" : "NO");
	clGetDeviceInfo(device, CL_DEVICE_LINKER_AVAILABLE, sizeof(cl_bool), &info_bool, NULL);
	printf("Наличие линковщика: %s\n", info_bool ? "YES" : "NO");
	clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &info_uint, NULL);
	printf("Макс. одновременно рабочих групп: %u\n", info_uint);
	clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &info_uint, NULL);
	printf("Макс. возможно измерений: %u\n", info_uint);
	clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &info_size_t, NULL);
	printf("Макс. возможно процессов в группе: %u\n", info_size_t);
	printf("\n");
	const size_t local_work_size = info_size_t; // Количество work-intem в локальной группе.
	const size_t global_work_size = local_work_size * ceil((double)points / local_work_size); // Всего work-intems.

	/* Шаг 3: Создание контекста.*/
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
	if(status != CL_SUCCESS)
	{
		printf("Ошибка! Создание контекста.\n");
		clean();
		return FAILURE;
	}

	/* Шаг 4: Создание очереди команд, ассоциированной с контекстом.*/
	commandQueue = clCreateCommandQueue(context, device, 0, &status);
	if(status != CL_SUCCESS)
	{
		printf("Ошибка! Создание очереди команд.\n");
		clean();
		return FAILURE;
	}

	/* Шаг 5: Создание программного объекта. */
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
		printf("Ошибка! Создание программного объекта.\n");
		clean();
		free(source);
		return FAILURE;
	}
	free(source);
	
	/* Шаг 6: Сборка программы. */
	status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	if(status != CL_SUCCESS)
	{
		printf("Ошибка! Сборка программы.\n");
		clean();
		return FAILURE;
	}

	/* Шаг 7: Инициализация входных, выходных буферов для хоста и ядра. */
	dataBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, (real_data_length) * sizeof(float), (void*) data, &status);
	if(status != CL_SUCCESS)
	{
		printf("Ошибка! Недостаточно видеопамяти для создания буфера данных.\n");
		clean();
		return FAILURE;
	}
	prevCentroidsBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR, clusters * n * sizeof(float), (void*) prev_centroids, &status);
	if(status != CL_SUCCESS)
	{
		printf("Ошибка! Недостаточно видеопамяти для создания буфера предыдущих центроидов.\n");
		clean();
		return FAILURE;
	}
	centroidsBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR, clusters * n * sizeof(float), (void*) centroids, &status);
	if(status != CL_SUCCESS)
	{
		printf("Ошибка! Недостаточно видеопамяти для создания буфера центроидов.\n");
		clean();
		return FAILURE;
	}
	pointClusterBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, points * sizeof(int), (void*) point_cluster, &status);
	if(status != CL_SUCCESS)
	{
		printf("Ошибка! Недостаточно видеопамяти для создания буфера точка->кластер.\n");
		clean();
		return FAILURE;
	}
	pointsInClBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR, clusters * sizeof(int), (void*) points_in_cluster, &status);
	if(status != CL_SUCCESS)
	{
		printf("Ошибка! Недостаточно видеопамяти для создания буфера количества точек в кластере.\n");
		clean();
		return FAILURE;
	}
	cMassBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR, clusters * n * sizeof(float), (void*) center_mass, &status);
	if(status != CL_SUCCESS)
	{
		printf("Ошибка! Недостаточно видеопамяти для создания буфера центра масс.\n");
		clean();
		return FAILURE;
	}

	/* Шаг 8: Создание ядра. */
	kernel = clCreateKernel(program, "kmeans", &status);
	if(status != CL_SUCCESS)
	{
		printf("Ошибка! Создание ядра.\n");
		clean();
		return FAILURE;
	}

	/* Шаг 9: Установка аргументов ядра.*/
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
		printf("Ошибка! Установка аргументов ядра.\n");
		clean();
		return FAILURE;
	}

	/* Шаг 10: Запуск ядра.*/
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
			printf("Ошибка! Запуск ядра.\n");
			clean();
			return FAILURE;
		}
		clock_t start_time = clock();
		status = clWaitForEvents(1, &event);
		if(status != CL_SUCCESS)
		{
			printf("Ошибка! Ожидание окончания работы ядра.\n");
			clean();
			return FAILURE;
		}
		//printf("Работа ядра успешно завершена.\n");
		//printf("Время работы ядра: %f сек.\n", ((float)(clock() - start_time))/CLOCKS_PER_SEC);

		// Сохраняем предыдущие центроиды.
		for(int i = 0; i < (clusters * n); i++)
			prev_centroids[i] = centroids[i];

		// Вычисляем новые координаты центроидов.
		for(int i = 0; i < clusters; i++)
			for(int j = i * n, k = 0; j < (i * n + n); j++, k++)
				centroids[j] = center_mass[j] / points_in_cluster[i];

		// Проверка совпадения текущих центров масс с предыдущими.
		int chk = 0;
		for(int i = 0; i < (n * clusters); i++)
			if(fabs(prev_centroids[i] - centroids[i]) <= eps)
				chk += 1;
		if(chk == (n * clusters))
			break;
	}

	printf("Время работы алгоритма: %f сек.\n", ((float)(clock() - alg_time))/CLOCKS_PER_SEC);

	status = clEnqueueReadBuffer(commandQueue, pointClusterBuffer, CL_TRUE, 0, points * sizeof(int), (void*) point_cluster, 0, NULL, NULL);
	if(status != CL_SUCCESS)
	{
		printf("Ошибка! Получение окончательных результатов работы от вычислителя.\n");
		clean();
		return FAILURE;
	}
	writeResult(centroids, n * clusters, data, real_data_length, n, point_cluster);

	clean();
	return SUCCESS;
}