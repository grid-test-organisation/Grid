    /*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid 

    Source file: ./benchmarks/Benchmark_mobius_fusion.cc

    Copyright (C) 2015

Author: Gianluca Filaci <g.filaci@ed.ac.uk>

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
#include <Grid/Grid.h>

using namespace std;
using namespace Grid;

int main (int argc, char ** argv)
{
  Grid_init(&argc,&argv);

  int threads = GridThread::GetThreads();
  std::cout<< GridLogMessage << "Grid is setup to use " << threads << " threads" << std::endl;
#ifdef GRID_NVCC
  std::cout<< GridLogMessage << "Grid is setup to use " << gpu_threads << " gpu-threads per block" << std::endl;
#endif
  
  Coordinate mpi_layout  = GridDefaultMpi();
  Coordinate default_latt = GridDefaultLatt();
  Coordinate latt4 ({default_latt[0]*mpi_layout[0], default_latt[1]*mpi_layout[1], default_latt[2]*mpi_layout[2], default_latt[3]*mpi_layout[3]});
  
  int Ls=16;
  for(int i=0;i<argc;i++)
    if(std::string(argv[i]) == "-Ls"){
      std::stringstream ss(argv[i+1]); ss >> Ls;
    }
  
  std::cout<< GridLogMessage << "Local volume : " << default_latt << " x " << Ls << std::endl;
  std::cout<< GridLogMessage << "Global volume: " << latt4        << " x " << Ls << std::endl;
  
  GridCartesian         * UGrid   = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(), GridDefaultSimd(Nd,vComplex::Nsimd()),GridDefaultMpi());
  GridRedBlackCartesian * UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  GridCartesian         * FGrid   = SpaceTimeGrid::makeFiveDimGrid(Ls,UGrid);
  GridRedBlackCartesian * FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls,UGrid);

  std::vector<int> seeds4({1,2,3,4});
  std::vector<int> seeds5({5,6,7,8});

  GridParallelRNG RNG4(UGrid); RNG4.SeedFixedIntegers(seeds4);
  std::cout << GridLogMessage << "Seeded" << std::endl;

  LatticeGaugeField Umu(UGrid); SU3::HotConfiguration(RNG4,Umu);
  std::cout << GridLogMessage << "made random gauge fields"<<std::endl;

  const int ncall=1000;
  double t0,t1;
  
  RealD mass=0.1;
  RealD M5  =1.8;
  
  GridParallelRNG RNG5(FGrid); RNG5.SeedFixedIntegers(seeds5);
  LatticeFermion src(FGrid); random(RNG5,src);
  LatticeFermion result(FGrid);
  
  LatticeFermion r_eo(FGrid);
  LatticeFermion src_e (FrbGrid);
  LatticeFermion src_o (FrbGrid);
  LatticeFermion r_e   (FrbGrid);
  LatticeFermion r_o   (FrbGrid);
  
  pickCheckerboard(Even,src_e,src);
  pickCheckerboard(Odd,src_o,src);
  
  setCheckerboard(r_eo,src_o);
  setCheckerboard(r_eo,src_e);
  
  r_e = Zero();
  r_o = Zero();

  auto ref = r_o;
  auto tmp_o = r_o;
  
  DomainWallFermionR Dw(Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5);
  //  MobiusFermionR Dw(Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,1.5,0.5);
  
#define PRINT(r,A) \
  if(latt4[0]==4) {\
    r_eo = Zero(); \
    setCheckerboard(r_eo,r); \
    std::cout << "--- RESULT Dw." #A " ---" <<std::endl; \
    std::cout << r_eo           <<std::endl; \
    std::cout << "--------------" <<std::endl; \
  }
  
#define REPORT true
#define NOREPORT false
  
#define BENCH_R(report,A,...)      \
  A(__VA_ARGS__);        \
  FGrid->Barrier();        \
  if(report) Dw.CayleyZeroCounters();      \
  t0=usecond();        \
  for(int i=0;i<ncall;i++){      \
    A(__VA_ARGS__);        \
  }            \
  t1=usecond();        \
  FGrid->Barrier();        \
  if(report) Dw.CayleyReport();          \
  std::cout << GridLogMessage << "$ Called " #A " "<< (t1-t0)/ncall << " us" << std::endl;\
  std::cout << GridLogMessage << "******************"<<std::endl;

#define BENCH_FUS_R(report,A,B,in,out,tmp)      \
  B(in,tmp);        \
  A(tmp,out);        \
  FGrid->Barrier();        \
  if(report) Dw.CayleyZeroCounters();      \
  t0=usecond();        \
  for(int i=0;i<ncall;i++){      \
    B(in,tmp);        \
    A(tmp,out);        \
  }            \
  t1=usecond();        \
  FGrid->Barrier();        \
  if(report) Dw.CayleyReport();          \
  std::cout << GridLogMessage << "$ Called " #A "+" #B " " << (t1-t0)/ncall << " us" << std::endl;\
  std::cout << GridLogMessage << "******************"<<std::endl;

#define BENCH(A,...) BENCH_R(REPORT,A,__VA_ARGS__)
#define BENCH_NOREP(A,...) BENCH_R(NOREPORT,A,__VA_ARGS__)
#define BENCH_FUS(A,...) BENCH_FUS_R(REPORT,A,__VA_ARGS__)
#define BENCH_FUS_NOREP(A,...) BENCH_FUS_R(NOREPORT,A,__VA_ARGS__)
  
#define COMPARE(r_o,ref) \
  r_o -= ref; \
  std::cout << GridLogMessage << "norm diff = " << norm2(r_o) << " (ref was " << norm2(ref) << ")" << std::endl; \
  std::cout<<GridLogMessage << "******************" << std::endl;

  std::cout << GridLogMessage<< "*********************************************************" <<std::endl;
  std::cout << GridLogMessage<< "* Benchmarking Cayley 5D functions"<<std::endl;
  std::cout << GridLogMessage<< "*********************************************************" <<std::endl;
  
  // pick some random coeffs
  typedef typename DomainWallFermionR::Coeff_t Coeff_t;
  Vector<Coeff_t> diag(Ls,1.);
  Vector<Coeff_t> upper(Ls,2.4);
  Vector<Coeff_t> lower(Ls,1.9);
  upper[Ls-1] = -Dw.mass*upper[Ls-1];
  lower[0]    = -Dw.mass*lower[0];

  Dw.M5D(src_o,src_o,r_o,lower,diag,upper);
  BENCH(Dw.M5D,src_o,src_o,r_o,lower,diag,upper);    PRINT(r_o,Dw.M5D);
  BENCH(Dw.M5Ddag,src_o,src_o,r_o,lower,diag,upper); PRINT(r_o,Dw.M5Ddag);
  BENCH(Dw.MooeeInv,src_o,r_o);                      PRINT(r_o,Dw.MooeeInv);
  BENCH(Dw.MooeeInvDag,src_o,r_o);                   PRINT(r_o,Dw.MooeeInvDag);
  
  std::cout << GridLogMessage<< "*********************************************************" <<std::endl;
  std::cout << GridLogMessage<< "* Benchmarking Cayley 5D FUSED functions"<<std::endl;
  std::cout << GridLogMessage<< "*********************************************************" <<std::endl;
  
  BENCH_FUS(Dw.Meooe5D,Dw.MooeeInv,src_o,r_o,tmp_o);
  ref = r_o;
  BENCH(Dw.Meooe5DMooeeInv,src_o,r_o);
  COMPARE(r_o,ref);

  BENCH_FUS(Dw.MooeeInvDag,Dw.MeooeDag5D,src_o,r_o,tmp_o);
  ref = r_o;
  BENCH(Dw.MooeeInvDagMeooeDag5D,src_o,r_o);
  COMPARE(r_o,ref);

  std::cout << GridLogMessage<< "*********************************************************" <<std::endl;
  std::cout << GridLogMessage<< "* Benchmarking fused kernels for RH preconditioning"<<std::endl;
  std::cout << GridLogMessage<< "*********************************************************" <<std::endl;
  
  SchurDiagOneRH<DomainWallFermionR,LatticeFermion>           SchurRH(Dw);
  SchurDiagOneRHFused<DomainWallFermionR,LatticeFermion> FusedSchurRH(Dw);
  
  std::cout << GridLogMessage<< "---------------------------------" <<std::endl;
  std::cout << GridLogMessage<< "-  Serial: comms then compute   -" <<std::endl;
  std::cout << GridLogMessage<< "---------------------------------" <<std::endl;
  WilsonKernelsStatic::Comms = WilsonKernelsStatic::CommsAndCompute;
  
  BENCH(SchurRH.Mpc,src_o,r_o);
  ref = r_o;
  BENCH(FusedSchurRH.Mpc,src_o,r_o);
  COMPARE(r_o,ref);

  BENCH_NOREP(SchurRH.MpcDag,src_o,r_o);
  ref = r_o;
  BENCH_NOREP(FusedSchurRH.MpcDag,src_o,r_o);
  COMPARE(r_o,ref);
  
  std::cout << GridLogMessage<< "---------------------------------" <<std::endl;
  std::cout << GridLogMessage<< "- Overlapped: comms and compute -" <<std::endl;
  std::cout << GridLogMessage<< "---------------------------------" <<std::endl;
  WilsonKernelsStatic::Comms = WilsonKernelsStatic::CommsThenCompute;
  
  BENCH(SchurRH.Mpc,src_o,r_o);
  ref = r_o;
  BENCH(FusedSchurRH.Mpc,src_o,r_o);
  COMPARE(r_o,ref);

  BENCH_NOREP(SchurRH.MpcDag,src_o,r_o);
  ref = r_o;
  BENCH_NOREP(FusedSchurRH.MpcDag,src_o,r_o);
  COMPARE(r_o,ref);
  
  Grid_finalize();
}
