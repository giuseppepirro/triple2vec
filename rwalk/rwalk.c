#include "rwalk.h"
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
//
//int main()
//{
//    int arr[]  = {1, 2, 3, 4};
//    int freq[] = {1, 5, 2, 1};
//    int i, n = sizeof(arr) / sizeof(arr[0]);
//
//    // Use a different seed value for every run.
//    srand(time(NULL));
//
//    // Let us generate 10 random numbers accroding to
//    // given distribution
//    for (i = 0; i < 1; i++)
//      printf("%d\n", myRand(arr, freq, n));
//
//    return 0;
//}

int *getNeighs(int const * allNeighs, int current_element, int num_neighs)
{
	int *result=malloc(num_neighs);

	for (int i=0; i<num_neighs;i++)
	{
		result[i]=allNeighs[current_element+1+i];
	}
		return result;

}


int *getWeights(int const * allWeights, int current_element, int num_neighs)
{
	int *result=malloc(num_neighs);

	for (int i=0; i<num_neighs;i++)
	{
		result[i]=allWeights[current_element+1+i];
	}
		return result;

}

int *multiplyWeights(double weights [], int size)
{
	int *result=malloc(size);;

	for (int i=0;i<size;i++)
	{
		result[i]=weights[i]*100;
	}
	return result;

}

// Utility function to find ceiling of r in arr[l..h]
int findCeil(int arr[], int r, int l, int h)
{
    int mid;
    while (l < h)
    {
         mid = l + ((h - l) >> 1);  // Same as mid = (l+h)/2
        (r > arr[mid]) ? (l = mid + 1) : (h = mid);
    }
    return (arr[l] >= r) ? l : -1;
}

int getBiasedNeigh(int const* arr, int freq[], int n, unsigned int private_seed)
{
    // Create and fill prefix array
    int prefix[n], i;
    prefix[0] = freq[0];
    for (i = 1; i < n; ++i)
        prefix[i] = prefix[i - 1] + freq[i];
    // prefix[n-1] is sum of all frequencies. Generate a random number
    // with value from 1 to this sum

    //int r=(rand_r(&private_seed) % prefix[n - 1])+1;
    //srand(private_seed);
    int r = (rand() % prefix[n - 1]) + 1;
    // Find index of ceiling of r in prefix array
    int indexc = findCeil(prefix, r, 0, n - 1);
    return arr[indexc];
}


void biased_random_walk(int const* ptr, int const* neighs, int const* weights, int n, int num_walks,
                 int num_steps, int seed, int nthread, int* walks) {
  if (nthread > 0) {
    omp_set_num_threads(nthread);
  }
#pragma omp parallel
  {
  //printf("Starting biased random walks");
   int thread_num = omp_get_thread_num();
   unsigned int private_seed = (unsigned int)(seed + thread_num);
#pragma omp for
    for (int i = 0; i < n; i++) {
      int offset, num_neighs;
      for (int walk = 0; walk < num_walks; walk++) {
        int curr = i;
        offset = i * num_walks * (num_steps + 1) + walk * (num_steps + 1);
        walks[offset] = i;
        for (int step = 0; step < num_steps; step++) {
          num_neighs = ptr[curr + 1] - ptr[curr];
          if (num_neighs > 0)
          {
            //get index of the current element
            int *neighbours_nodes=getNeighs(neighs, curr, num_neighs);
            int * weights_neighbours_nodes=getWeights(weights, curr, num_neighs);
            curr = neighs[ptr[curr] + getBiasedNeigh(neighbours_nodes, weights_neighbours_nodes, num_neighs,private_seed)];
          }
          walks[offset + step + 1] = curr;
        }
      }
    }
  }
}



void random_walk(int const* ptr, int const* neighs, int n, int num_walks,
                 int num_steps, int seed, int nthread, int* walks) {
  if (nthread > 0) {
    omp_set_num_threads(nthread);
  }
#pragma omp parallel
  {
    int thread_num = omp_get_thread_num();
    unsigned int private_seed = (unsigned int)(seed + thread_num);
#pragma omp for
    for (int i = 0; i < n; i++) {
      int offset, num_neighs;
      for (int walk = 0; walk < num_walks; walk++) {
        int curr = i;
        offset = i * num_walks * (num_steps + 1) + walk * (num_steps + 1);
        walks[offset] = i;
        for (int step = 0; step < num_steps; step++) {
          num_neighs = ptr[curr + 1] - ptr[curr];
          if (num_neighs > 0)
          {
          //myRand myRand(arr, freq, n)
            curr = neighs[ptr[curr] + (rand_r(&private_seed) % num_neighs)];
          }
          walks[offset + step + 1] = curr;
        }
      }
    }
  }
}
