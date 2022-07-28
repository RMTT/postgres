#include "stdint.h"
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
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

void cuda_sort_int(int *data,SortTuple *index, int length) {
  thrust::device_vector<int> d_array(data, data + length);
  thrust::device_vector<SortTuple> d_index(index, index + length);

  thrust::sort_by_key(thrust::device, d_array.begin(), d_array.end(), d_index.begin());

  thrust::copy(d_index.begin(), d_index.end(), index);
}
}
