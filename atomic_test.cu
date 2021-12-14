#include <cuda_runtime.h>                                                       
#include <cuda.h>                                                               
#include <curand_kernel.h>                                                      
#include <stdlib.h>                                                             
#include <stdio.h>                                                              


__global__ void test_atomics(int* lock, int* accumulator){
  int res = 1;
  do {
    res = atomicCAS(lock, 0, 1);
    if (res == 0) {
      (*accumulator)++;
      atomicExch(lock, 0);
    }
  } while (res == 1);
}                                                                               
                                                                                
int main(int argc, char** argv) {                                               
    printf("Start\n");
    int* h_lock = new int;                                                      
    int* h_accumulator = new int;                                               
    int* d_lock;                                                                
    int* d_accumulator;                                                         
                                                                                
                                                                                
    *h_lock = 0;                                                                
    *h_accumulator = 0;                                                         
                                                                                
    //copy to device                                                            
    cudaMalloc((void**)&d_lock, sizeof(int));                                   
    cudaMemcpy(d_lock, h_lock, sizeof(int), cudaMemcpyHostToDevice);            
    cudaMalloc((void**)&d_accumulator, sizeof(int));                            
    cudaMemcpy(d_accumulator, h_lock, sizeof(int), cudaMemcpyHostToDevice);     
    printf("run kernel\n");                                                                                
    //run kernel                                                                
    test_atomics<<<1,32>>>(d_lock, d_accumulator);
    cudaDeviceSynchronize();
    printf("kernel done\n");
    
    //copy to host                                                              
    cudaMemcpy(h_lock, d_lock, sizeof(int), cudaMemcpyDeviceToHost);            
    cudaMemcpy(h_accumulator, d_accumulator, sizeof(int), cudaMemcpyDeviceToHost);
                                                                                
    //print result                                                              
    printf("lock: %d, accumulator: %d\n", *h_lock, *h_accumulator);             
                                                                                
                                                                                
    cudaFree(d_lock);                                                           
    cudaFree(d_accumulator);                                                    
    delete h_lock;                                                              
    delete h_accumulator;                                                       
    return 0;                                                                   
} 
