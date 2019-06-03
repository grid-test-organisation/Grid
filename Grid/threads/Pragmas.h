/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid 

    Source file: ./lib/Threads.h

    Copyright (C) 2015

Author: Peter Boyle <paboyle@ph.ed.ac.uk>
Author: paboyle <paboyle@ph.ed.ac.uk>

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

    See the full license in the file "LICENSE" in the top level distribution directory
*************************************************************************************/
/*  END LEGAL */
#pragma once

#ifndef MAX
#define MAX(x,y) ((x)>(y)?(x):(y))
#define MIN(x,y) ((x)>(y)?(y):(x))
#endif

#define strong_inline     __attribute__((always_inline)) inline

#ifdef _OPENMP
#define GRID_OMP
#include <omp.h>
#endif

#ifdef __NVCC__
#define GRID_NVCC
#endif

#define UNROLL  _Pragma("unroll")

//////////////////////////////////////////////////////////////////////////////////
// New primitives; explicit host thread calls, and accelerator data parallel calls
//////////////////////////////////////////////////////////////////////////////////
#ifdef GRID_OMP

#define DO_PRAGMA_(x) _Pragma (#x)
#define DO_PRAGMA(x) DO_PRAGMA_(x)

#define thread_loop( range , ... )                        DO_PRAGMA(omp parallel for schedule(static))for range { __VA_ARGS__ };
#define thread_loop_collapse2( range , ... )              DO_PRAGMA(omp parallel for collapse(2))     for range { __VA_ARGS__ };
#define thread_loop_collapse( N , range , ... )           DO_PRAGMA(omp parallel for collapse ( N ) ) for range { __VA_ARGS__ };
#define thread_loop_in_region( range , ... )              DO_PRAGMA(omp for schedule(static))         for range { __VA_ARGS__ };
#define thread_loop_collapse_in_region( N , range , ... ) DO_PRAGMA(omp for collapse ( N ))           for range { __VA_ARGS__ };
#define thread_region                                     DO_PRAGMA(omp parallel)
#define thread_critical                                   DO_PRAGMA(omp critical)
#define thread_num(a) omp_get_thread_num()
#define thread_max(a) omp_get_max_threads()
#else
#define thread_loop( range , ... )                for range { __VA_ARGS__ ; };
#define thread_loop_collapse2( range , ... )      for range { __VA_ARGS__ ; };
#define thread_loop_collapse( N , range , ... )  for range { __VA_ARGS__ ; };
#define thread_region                           
#define thread_loop_in_region( range , ... )  for range { __VA_ARGS__ ; };
#define thread_loop_collapse_in_region( N, range , ... ) for range  { __VA_ARGS__ ; };

#define thread_critical                         
#define thread_num(a) (0)
#define thread_max(a) (1)
#endif


//////////////////////////////////////////////////////////////////////////////////
// Accelerator primitives; fall back to threading
//////////////////////////////////////////////////////////////////////////////////
#ifdef GRID_NVCC

extern uint32_t gpu_threads;

template<typename lambda>  __global__
void LambdaApply(uint64_t base, uint64_t Num, lambda Lambda)
{
  uint64_t ss = blockIdx.x*blockDim.x + threadIdx.x;
  if ( ss < Num ) {
    Lambda(ss+base);
  }
}

#define accelerator        __host__ __device__
#define accelerator_inline __host__ __device__ inline
#define accelerator_loop( iterator, range, ... )			\
  typedef decltype(range.begin()) Iterator;				\
  auto lambda = [=] accelerator (Iterator iterator) mutable {		\
    __VA_ARGS__;							\
  };									\
  Iterator num  = range.end() - range.begin();				\
  Iterator base = range.begin();					\
  Iterator num_block  = (num+gpu_threads-1)/gpu_threads;		\
  LambdaApply<<<num_block,gpu_threads>>>(base,num,lambda);		\
  cudaDeviceSynchronize();						\
  cudaError err = cudaGetLastError();					\
  if ( cudaSuccess != err ) {						\
    printf("Cuda error %s\n",cudaGetErrorString( err ));		\
    exit(0);								\
  }									

#define accelerator_loopN( iterator, num, ... )			\
  typedef decltype(num) Iterator;				\
  if ( num > 0 ) {			                        \
    auto lambda = [=] accelerator (Iterator iterator) mutable { \
      __VA_ARGS__;						\
    };								\
    Iterator base = 0;						\
    Iterator num_block  = (num+gpu_threads-1)/gpu_threads;	\
    LambdaApply<<<num_block,gpu_threads>>>(base,num,lambda);	\
    cudaDeviceSynchronize();					\
    cudaError err = cudaGetLastError();				\
    if ( cudaSuccess != err ) {					\
      printf("Cuda error %s\n",cudaGetErrorString( err ));	\
      exit(0);							\
    }								\
  }

#define accelerator_loopNB( iterator, num, ... )			\
  typedef decltype(num) Iterator;				\
  if ( num > 0 ) {			                        \
    auto lambda = [=] accelerator (Iterator iterator) mutable { \
      __VA_ARGS__;						\
    };								\
    Iterator base = 0;						\
    Iterator num_block  = (num+gpu_threads-1)/gpu_threads;	\
    LambdaApply<<<num_block,gpu_threads>>>(base,num,lambda);	\
  }

#define cpu_loop( iterator, range, ... )   thread_loop( (auto iterator = range.begin();iterator<range.end();iterator++), { __VA_ARGS__ });

template<typename lambda>  __global__
void LambdaApply2D(uint64_t Osites, uint64_t Isites, lambda Lambda)
{
  uint64_t site  = threadIdx.x + blockIdx.x*blockDim.x;
  uint64_t osite = site / Isites;
  if ( (osite <Osites) ) {
    Lambda(osite);
  }
}

#define coalesce_loop( iterator, range, nsimd, ... )			\
  typedef uint64_t Iterator;						\
  auto lambda = [=] accelerator (Iterator iterator) mutable {		\
    __VA_ARGS__;							\
  };									\
  Iterator num  = range.end() - range.begin();				\
  Iterator base = range.begin();					\
  Iterator cu_threads= gpu_threads;					\
  Iterator cu_blocks = num*nsimd/cu_threads;				\
  LambdaApply2D<<<cu_blocks,cu_threads>>>(num,(uint64_t)nsimd,lambda);	\
  cudaDeviceSynchronize();						\
  cudaError err = cudaGetLastError();					\
  if ( cudaSuccess != err ) {						\
    printf("Cuda error %s\n",cudaGetErrorString( err ));		\
    exit(0);								\
  }									

#else

#define accelerator 
#define accelerator_inline strong_inline
#define accelerator_loop( iterator, range, ... )			\
  thread_loop( (auto iterator = range.begin();iterator<range.end();iterator++), { __VA_ARGS__ });

#define accelerator_loopN( iterator, num, ... )			\
  thread_loop( (auto iterator = 0;iterator<num;iterator++), { __VA_ARGS__ });

#define cpu_loop( iterator, range, ... )				\
  thread_loop( (auto iterator = range.begin();iterator<range.end();iterator++), { __VA_ARGS__ });

#endif
