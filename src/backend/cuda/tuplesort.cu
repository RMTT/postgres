#include "stdint.h"
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <cstring>

struct SortTuple {
  void *tuple;
  int datum1;
  bool isnull1;
  int srctape;
};

__host__ __device__ bool operator<(const SortTuple &a, const SortTuple &b) {
  return a.datum1 < b.datum1;
}

extern "C" {
void cuda_sort(SortTuple *memtuples, int length) {
  thrust::device_vector<SortTuple> tuples(memtuples, memtuples + length);

  thrust::sort(thrust::device, tuples.begin(), tuples.end());

  thrust::copy(tuples.begin(), tuples.end(), memtuples);
}


void cuda_sort_int(int *data, int length) {
    thrust::device_vector<unsigned int> d_array(data, data + length);
    thrust::sort(thrust::device, d_array.begin(), d_array.end());
}
}
