void atomic_add_global(volatile global float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;
 
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((volatile global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

__kernel void kmeans(__global float* data, __global float* prev_centroids, __global float* centroids, __global int* point_cluster, __global int* points_in_cluster, __global float* center_mass, int n, int clusters, int points)
{
	int global_id = get_global_id(0);
	if(global_id >= points)
		return;

	float min = -1, tmp, sum;
	int min_i = 0;

	// Определение ближайшего к точке центроида.
	for(int i = 0; i < clusters; i += 1)
	{
		sum = 0;
		tmp = 0;
		for(int j = i * n, k = global_id * n; j < (i * n + n); j += 1, k += 1)
			sum += pow((centroids[j] - data[k]), 2);
		tmp = sqrt(sum);
		if(min == -1)
			min = tmp;
		else
		{
			if(min > tmp)
			{
				min = tmp;
				min_i = i;
			}
		}
	}

	// Привязка точки к центроиду.
	point_cluster[global_id] = min_i;
	atomic_add(&points_in_cluster[min_i], 1);

	// Подсчет центра масс.
	for(int i = global_id * n, j = 0; i < (global_id * n + n); i += 1, j += 1)
		atomic_add_global(&center_mass[min_i * n + j], data[i]);
}