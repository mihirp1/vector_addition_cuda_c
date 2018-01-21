#include<stdio.h>
#include<stdlib.h>
#include <sys/time.h>
#define imin(a,b) (a<b?a:b)

const int N = 16777216;
const int TH_B = 512;
const int blocksPerGrid = imin( 32, (N+TH_B-1) / TH_B );


long long start_timer() {

struct timeval tv;

gettimeofday(&tv, NULL);

return tv.tv_sec * 1000000 + tv.tv_usec;

}


long long stop_timer(long long start_time,char *name) {

struct timeval tv;

gettimeofday(&tv, NULL);

long long end_time = tv.tv_sec * 1000000 + tv.tv_usec;

float total_time = (end_time - start_time)/1000000.0;

printf("Value of Blocks Per Grid : %d",blocksPerGrid);

//print execution time for cpu
if(name=="Cpu")
{

printf("\nC) (T%s) Execution Time for Serial Algorithm or %s : %.5f sec\n",name,name,total_time);
}

//print execution time for gpu and kernel time
if(name=="Gpu")
{
printf("\nE) Kernel execution Time is %.5f sec\n",total_time);
printf("\nF) (T%s) Execution Time for Parallel Algorithm or %s :  %.5f sec\n",name,name,total_time);
}

//print execution time for memory allocation in gpu
if(name=="memalloctgpu")
{

printf("\nB) Memory allocation Time for GPU is : %.5f sec\n",total_time);

}

//print execution time for memory allocation in cpu
if(name=="memalloctcpu")
{

printf("\nA) Memory allocation Time for CPU is : %.5f sec\n",total_time);

}


//print condition for cpu to gpu time

if(name=="c2g")
{
printf("\nD) Data Transfer from CPU to GPU time is : %.5f sec\n",total_time);

}
//print condition for gpu to cpu transfer time
if(name=="g2c")
{

printf("\nG) Data Transfer from GPU to CPU time is : %.5f sec\n",total_time);

}

return ((end_time) - (start_time));

}


__global__ void GPU_big_dot( float *a, float *b, float *c ) {
    __shared__ float cache[TH_B];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float   temp = 0;
    while (tid < N) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    // assign the cache 
    cache[cacheIndex] = temp;

    // synchronize threads in this block
    __syncthreads();

    int i = blockDim.x/2;
    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0)
        c[blockIdx.x] = cache[0];


}

float CPU_big_dot(float *a,float *b)
{
              float  cpu_sum;
              for(int tid =0;tid<N;tid++)
              {
                      a[tid]=tid;
                      b[tid]= tid * 2;
                      cpu_sum += a[tid] * b[tid];
               }
              return cpu_sum;
}


int main( void ) {

   long long s_t;
   long long s_tt;
   long long s_tt_g2c;
   long long s_t_c2g;
   long long cpu_i;
   long long gpu_i;
   float spu;
   float CPU_SUM;
   long long s_t_cpu_memalloc;
   long long s_t_gpu_memalloc;


    float   *a, *b, c, *partial_c;
    float   *d_a, *d_b, *d_partial_c;

    s_t_cpu_memalloc = start_timer();

    // allocate memory on the cpu side
    a = (float*)malloc( N*sizeof(float) );
    b = (float*)malloc( N*sizeof(float) );
    partial_c = (float*)malloc( blocksPerGrid*sizeof(float) );
    
    stop_timer(s_t_cpu_memalloc,(char*)"memalloctcpu");

    s_t_gpu_memalloc = start_timer();


    // allocate the memory on the GPU
    cudaMalloc( (void**)&d_a,
                              N*sizeof(float) ) ;
    cudaMalloc( (void**)&d_b,
                              N*sizeof(float) ) ;
    cudaMalloc( (void**)&d_partial_c,
                              blocksPerGrid*sizeof(float) ) ;

    stop_timer(s_t_gpu_memalloc,(char*)"memalloctgpu");


    //fill in the host memory with data
    for (int i=0; i<N; i++) {
        a[i] = i;
        //b[i] = i * 2;
         b[i] = i;
    }

   s_t = start_timer();
    CPU_SUM = CPU_big_dot(a,b);
    cpu_i=stop_timer(s_t,(char*)"Cpu");
 

    s_t_c2g = start_timer();
     
    
    // copy the arrays 'a' and 'b' to the GPU
    cudaMemcpy( d_a, a, N*sizeof(float),
                              cudaMemcpyHostToDevice ) ;
    cudaMemcpy( d_b, b, N*sizeof(float),
                              cudaMemcpyHostToDevice ) ;

    stop_timer(s_t_c2g,(char*)"c2g");

  s_tt = start_timer();
  GPU_big_dot<<<blocksPerGrid,TH_B>>>( d_a, d_b,
                                            d_partial_c );


  gpu_i=stop_timer(s_tt, (char*)"Gpu");
    // copy the array 'c' back from the GPU to the CPU

   s_tt_g2c = start_timer();
   
    cudaMemcpy( partial_c, d_partial_c,
                              blocksPerGrid*sizeof(float),
                              cudaMemcpyDeviceToHost ) ;
   stop_timer(s_tt_g2c, (char*)"g2c");



  spu=(float)((float)cpu_i/(float)gpu_i);
  printf("\nH) Total SpeedUp is : %f \n",spu);

    // finish up on the CPU side
                                        
   c = 0;
    for (int i=0; i<blocksPerGrid; i++) {
        c += partial_c[i];
    }

    printf( "\nI) GPU dot-product value is %f = %.6g\n", c,c);


    printf( "\nJ) CPU dot-product value is %f = %.6g\n\n", CPU_SUM,CPU_SUM );


    // free memory on the gpu side
    cudaFree( d_a ) ;
    cudaFree( d_b ) ;
    cudaFree( d_partial_c ) ;

    // free memory on the cpu side
    free( a );
    free( b );
    free( partial_c );
}

