/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Source file: ./lib/algorithms/iterative/ConjugateGradient.h

Copyright (C) 2015

Author: Azusa Yamaguchi <ayamaguc@staffmail.ed.ac.uk>
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

See the full license in the file "LICENSE" in the top level distribution
directory
*************************************************************************************/
			   /*  END LEGAL */
#ifndef GRID_CONJUGATE_GRADIENT_H
#define GRID_CONJUGATE_GRADIENT_H

NAMESPACE_BEGIN(Grid);

/////////////////////////////////////////////////////////////
// Base classes for iterative processes based on operators
// single input vec, single output vec.
/////////////////////////////////////////////////////////////

template <class Field>
class ConjugateGradient : public OperatorFunction<Field> {
public:

  using OperatorFunction<Field>::operator();

  bool ErrorOnNoConverge;  // throw an assert when the CG fails to converge.
                           // Defaults true.
  RealD Tolerance;
  Integer MaxIterations;
  Integer IterationsToComplete; //Number of iterations the CG took to finish. Filled in upon completion
  
  ConjugateGradient(RealD tol, Integer maxit, bool err_on_no_conv = true)
    : Tolerance(tol),
      MaxIterations(maxit),
      ErrorOnNoConverge(err_on_no_conv){};

  void operator()(LinearOperatorBase<Field> &Linop, const Field &src, Field &psi) {

    psi.Checkerboard() = src.Checkerboard();

    conformable(psi, src);

    RealD cp, c, a, d, b, ssq, qq;
    //RealD b_pred;

    Field p(src.Grid());
    Field mmp(src.Grid());
    Field r(src.Grid());
    
    // Initial residual computation & set up
    RealD guess = norm2(psi);
    assert(std::isnan(guess) == 0);

    
    Linop.HermOpAndNorm(psi, mmp, d, b);
    
    r = src - mmp;
    p = r;

    a = norm2(p);
    cp = a;
    ssq = norm2(src);

    std::cout << GridLogIterative << std::setprecision(8) << "ConjugateGradient: guess " << guess << std::endl;
    std::cout << GridLogIterative << std::setprecision(8) << "ConjugateGradient:   src " << ssq << std::endl;
    std::cout << GridLogIterative << std::setprecision(8) << "ConjugateGradient:    mp " << d << std::endl;
    std::cout << GridLogIterative << std::setprecision(8) << "ConjugateGradient:   mmp " << b << std::endl;
    std::cout << GridLogIterative << std::setprecision(8) << "ConjugateGradient:  cp,r " << cp << std::endl;
    std::cout << GridLogIterative << std::setprecision(8) << "ConjugateGradient:     p " << a << std::endl;

    RealD rsq = Tolerance * Tolerance * ssq;

    // Check if guess is really REALLY good :)
    if (cp <= rsq) {
      return;
    }

    std::cout << GridLogIterative << std::setprecision(8)
              << "ConjugateGradient: k=0 residual " << cp << " target " << rsq << std::endl;

    GridStopWatch LinalgTimer;
    GridStopWatch InnerTimer;
    GridStopWatch AxpyNormTimer;
    GridStopWatch LinearCombTimer;
    GridStopWatch MatrixTimer;
    GridStopWatch SolverTimer;

    SolverTimer.Start();
    int k;
    for (k = 1; k <= MaxIterations*1000; k++) {
      c = cp;

      MatrixTimer.Start();
      Linop.HermOp(p, mmp);
      MatrixTimer.Stop();

      LinalgTimer.Start();

      InnerTimer.Start();
      ComplexD dc  = innerProduct(p,mmp);
      InnerTimer.Stop();
      d = dc.real();
      a = c / d;

      AxpyNormTimer.Start();
      cp = axpy_norm(r, -a, mmp, r);
      AxpyNormTimer.Stop();
      b = cp / c;

      LinearCombTimer.Start();
      auto psi_v = psi.View();
      auto p_v   = p.View();
      auto r_v   = r.View();
#ifdef GRID_NVCC
      const uint64_t nsimd = psi.Grid()->Nsimd();
      const uint64_t sites = nsimd * psi.Grid()->oSites();
      accelerator_loopN(sss,sites,{
        uint64_t lane = sss % nsimd;
        uint64_t ss   = sss / nsimd;
        auto psi_l = extractLane(lane,psi_v[ss]);
        auto p_l   = extractLane(lane,p_v  [ss]);
        auto r_l   = extractLane(lane,r_v  [ss]);
        psi_l = a*p_l + psi_l;
        p_l   = b*p_l + r_l;
        insertLane(lane,psi_v[ss],psi_l);
        insertLane(lane,p_v  [ss],p_l  );
      });
#else
      accelerator_loop(ss,p_v,{
	vstream(psi_v[ss], a      *  p_v[ss] + psi_v[ss]);
	vstream(p_v  [ss], b      *  p_v[ss] + r_v[ss]);
      });
#endif
      LinearCombTimer.Stop();
      LinalgTimer.Stop();

      std::cout << GridLogIterative << "ConjugateGradient: Iteration " << k
                << " residual^2 " << sqrt(cp/ssq) << " target " << Tolerance << std::endl;

      // Stopping condition
      if (cp <= rsq) {
        SolverTimer.Stop();
        Linop.HermOpAndNorm(psi, mmp, d, qq);
        p = mmp - src;

        RealD srcnorm = std::sqrt(norm2(src));
        RealD resnorm = std::sqrt(norm2(p));
        RealD true_residual = resnorm / srcnorm;

        std::cout << GridLogMessage << "ConjugateGradient Converged on iteration " << k << std::endl;
        std::cout << GridLogMessage << "\tComputed residual " << std::sqrt(cp / ssq)<<std::endl;
	std::cout << GridLogMessage << "\tTrue residual " << true_residual<<std::endl;
	std::cout << GridLogMessage << "\tTarget " << Tolerance << std::endl;

        std::cout << GridLogMessage << "Time breakdown "<<std::endl;
	std::cout << GridLogMessage << "\tElapsed    " << SolverTimer.Elapsed() <<std::endl;
	std::cout << GridLogMessage << "\tMatrix     " << MatrixTimer.Elapsed() <<std::endl;
	std::cout << GridLogMessage << "\tLinalg     " << LinalgTimer.Elapsed() <<std::endl;
	std::cout << GridLogMessage << "\tInner      " << InnerTimer.Elapsed() <<std::endl;
	std::cout << GridLogMessage << "\tAxpyNorm   " << AxpyNormTimer.Elapsed() <<std::endl;
	std::cout << GridLogMessage << "\tLinearComb " << LinearCombTimer.Elapsed() <<std::endl;

        if (ErrorOnNoConverge) assert(true_residual / Tolerance < 10000.0);

	IterationsToComplete = k;	

        return;
      }
    }
    std::cout << GridLogMessage << "ConjugateGradient did NOT converge"
              << std::endl;

    if (ErrorOnNoConverge) assert(0);
    IterationsToComplete = k;

  }
};
NAMESPACE_END(Grid);
#endif
