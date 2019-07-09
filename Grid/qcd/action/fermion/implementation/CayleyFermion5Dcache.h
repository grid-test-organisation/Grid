/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid 

    Source file: ./lib/qcd/action/fermion/CayleyFermion5D.cc

    Copyright (C) 2015

Author: Peter Boyle <pabobyle@ph.ed.ac.uk>
Author: Peter Boyle <paboyle@ph.ed.ac.uk>
Author: Peter Boyle <peterboyle@Peters-MacBook-Pro-2.local>
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

#include <Grid/qcd/action/fermion/FermionCore.h>
#include <Grid/qcd/action/fermion/CayleyFermion5D.h>


NAMESPACE_BEGIN(Grid);

/******************************/
/*     5D INNER FUNCTIONS     */
/******************************/

template<class FermionFieldView, class CoeffsPtr>
accelerator_inline void M5DInner(int ss, int Ls, const FermionFieldView &psi,
                                 const FermionFieldView &phi,
                                 FermionFieldView &chi,
                                 CoeffsPtr lower,
                                 CoeffsPtr diag,
                                 CoeffsPtr upper)
{
  typedef decltype(coalescedRead(psi[0])) spinor;
  spinor tmp1, tmp2;
  for(int s=0;s<Ls;s++){
    uint64_t idx_u = ss+((s+1)%Ls);
    uint64_t idx_l = ss+((s+Ls-1)%Ls);
    spProj5m(tmp1,psi(idx_u));
    spProj5p(tmp2,psi(idx_l));
    // summing phi + P- first and then adding P+ is faster on GPU
    coalescedWrite(chi[ss+s],(diag[s]*phi(ss+s)+upper[s]*tmp1)+lower[s]*tmp2);
  }
}

template<class FermionFieldView, class CoeffsPtr>
accelerator_inline void M5DdagInner(int ss, int Ls, const FermionFieldView &psi,
                                 const FermionFieldView &phi,
                                 FermionFieldView &chi,
                                 CoeffsPtr lower,
                                 CoeffsPtr diag,
                                 CoeffsPtr upper)
{
  typedef decltype(coalescedRead(psi[0])) spinor;
  spinor tmp1, tmp2;
  for(int s=0;s<Ls;s++){
    uint64_t idx_u = ss+((s+1)%Ls);
    uint64_t idx_l = ss+((s+Ls-1)%Ls);
    spProj5p(tmp1,psi(idx_u));
    spProj5m(tmp2,psi(idx_l));
    // summing phi + P- first and then adding P+ is faster on GPU
    coalescedWrite(chi[ss+s],(diag[s]*phi(ss+s)+lower[s]*tmp2)+upper[s]*tmp1);
  }
}

template<class FermionFieldView, class CoeffsPtr>
accelerator_inline void MooeeInvInner (const int ss, const int Ls,
                                       const FermionFieldView &in_v, FermionFieldView &out_v,
                                       const CoeffsPtr dee_v,
                                       const CoeffsPtr lee_v, const CoeffsPtr leem_v,
                                       const CoeffsPtr uee_v, const CoeffsPtr ueem_v)
{
  typedef decltype(coalescedRead(in_v[0])) spinor;
  spinor tmp, acc, res;
  
  // X = Nc*Ns
  // flops = 2X + (Ls-2)(4X + 4X) + 6X + 1 + 2X + (Ls-1)(10X + 1) = -16X + Ls(1+18X) = -192 + 217*Ls flops
  // Apply (L^{\prime})^{-1} L_m^{-1}
  res = in_v(ss);
  spProj5m(tmp,res);
  acc = leem_v[0]*tmp;
  spProj5p(tmp,res);
  coalescedWrite(out_v[ss],res);
  
  for(int s=1;s<Ls-1;s++){
    res = in_v(ss+s);
    res -= lee_v[s-1]*tmp;
    spProj5m(tmp,res);
    acc += leem_v[s]*tmp;
    spProj5p(tmp,res);
    coalescedWrite(out_v[ss+s],res);
  }
  res = in_v(ss+Ls-1) - lee_v[Ls-2]*tmp - acc;
  
  // Apply U_m^{-1} D^{-1} U^{-1}
  res = (1.0/dee_v[Ls-1])*res;
  coalescedWrite(out_v[ss+Ls-1],res);
  spProj5p(acc,res);
  spProj5m(tmp,res);
  for (int s=Ls-2;s>=0;s--){
    res = (1.0/dee_v[s])*out_v(ss+s) - uee_v[s]*tmp - ueem_v[s]*acc;
    spProj5m(tmp,res);
    coalescedWrite(out_v[ss+s],res);
  }
}

template<class FermionFieldView, class CoeffsPtr>
accelerator_inline void MooeeInvDagInner (const int ss, const int Ls,
                                          const FermionFieldView &in_v, FermionFieldView &out_v,
                                          const CoeffsPtr dee_v,
                                          const CoeffsPtr lee_v, const CoeffsPtr leem_v,
                                          const CoeffsPtr uee_v, const CoeffsPtr ueem_v)
{
  typedef decltype(coalescedRead(in_v[0])) spinor;
  spinor tmp, acc, res;
  
  // X = Nc*Ns
  // flops = 2X + (Ls-2)(4X + 4X) + 6X + 1 + 2X + (Ls-1)(10X + 1) = -16X + Ls(1+18X) = -192 + 217*Ls flops
  // Apply (L^{\prime})^{-1} L_m^{-1}
  res = in_v(ss);
  spProj5p(tmp,res);
  acc = conjugate(ueem_v[0])*tmp;
  spProj5m(tmp,res);
  coalescedWrite(out_v[ss],res);
  
  for(int s=1;s<Ls-1;s++){
    res = in_v(ss+s);
    res -= conjugate(uee_v[s-1])*tmp;
    spProj5p(tmp,res);
    acc += conjugate(ueem_v[s])*tmp;
    spProj5m(tmp,res);
    coalescedWrite(out_v[ss+s],res);
  }
  res = in_v(ss+Ls-1) - conjugate(uee_v[Ls-2])*tmp - acc;
  
  // Apply U_m^{-1} D^{-1} U^{-1}
  res = (1.0/dee_v[Ls-1])*res;
  coalescedWrite(out_v[ss+Ls-1],res);
  spProj5m(acc,res);
  spProj5p(tmp,res);
  for (int s=Ls-2;s>=0;s--){
    res = (1.0/dee_v[s])*out_v(ss+s) - conjugate(lee_v[s])*tmp - conjugate(leem_v[s])*acc;
    spProj5p(tmp,res);
    coalescedWrite(out_v[ss+s],res);
  }
}

/*****************************/
/*       5D FUNCTIONS       */
/****************************/

template<class Impl>  
void
CayleyFermion5D<Impl>::M5D(const FermionField &psi_i,
			       const FermionField &phi_i, 
			       FermionField &chi_i,
			       Vector<Coeff_t> &lower,
			       Vector<Coeff_t> &diag,
			       Vector<Coeff_t> &upper)
{
  
  chi_i.Checkerboard()=psi_i.Checkerboard();
  GridBase *grid=psi_i.Grid();
  auto psi = psi_i.View();
  auto phi = phi_i.View();
  auto chi = chi_i.View();
  auto diag_v  = &diag[0];
  auto upper_v = &upper[0];
  auto lower_v = &lower[0];
  assert(phi.Checkerboard() == psi.Checkerboard());

  int Ls =this->Ls;

  // 10 = 3 complex mult + 2 complex add
  // Flops = 10.0*(Nc*Ns) *Ls*vol (/2 for red black counting)
  M5Dcalls++;
  M5Dtime-=usecond();

  uint64_t nloop = grid->oSites()/Ls;
  accelerator_for(sss,nloop,Simd::Nsimd(),{
    uint64_t ss= sss*Ls;
    M5DInner(ss,Ls,psi,phi,chi,lower_v,diag_v,upper_v);
  });
  M5Dtime+=usecond();
}

template<class Impl>
void
CayleyFermion5D<Impl>::M5Ddag(const FermionField &psi_i,
                           const FermionField &phi_i,
                           FermionField &chi_i,
                           Vector<Coeff_t> &lower,
                           Vector<Coeff_t> &diag,
                           Vector<Coeff_t> &upper)
{
  
  chi_i.Checkerboard()=psi_i.Checkerboard();
  GridBase *grid=psi_i.Grid();
  auto psi = psi_i.View();
  auto phi = phi_i.View();
  auto chi = chi_i.View();
  auto diag_v  = &diag[0];
  auto upper_v = &upper[0];
  auto lower_v = &lower[0];
  assert(phi.Checkerboard() == psi.Checkerboard());
  
  int Ls =this->Ls;
  
  // 10 = 3 complex mult + 2 complex add
  // Flops = 10.0*(Nc*Ns) *Ls*vol (/2 for red black counting)
  M5Dcalls++;
  M5Dtime-=usecond();
  
  uint64_t nloop = grid->oSites()/Ls;
  accelerator_for(sss,nloop,Simd::Nsimd(),{
    uint64_t ss= sss*Ls;
    M5DdagInner(ss,Ls,psi,phi,chi,lower_v,diag_v,upper_v);
  });
  M5Dtime+=usecond();
}

template<class Impl>
void
CayleyFermion5D<Impl>::MooeeInv    (const FermionField &psi_i, FermionField &chi_i)
{
  chi_i.Checkerboard()=psi_i.Checkerboard();
  GridBase *grid=psi_i.Grid();

  auto psi = psi_i.View();
  auto chi = chi_i.View();
  auto lee_v  = &lee[0];
  auto leem_v = &leem[0];
  auto uee_v  = &uee[0];
  auto ueem_v = &ueem[0];
  auto dee_v  = &dee[0];
  int Ls=this->Ls;

  MooeeInvCalls++;
  MooeeInvTime-=usecond();
  uint64_t nloop = grid->oSites()/Ls;
  accelerator_for(sss,nloop,Simd::Nsimd(),{
    uint64_t ss=sss*Ls;
    MooeeInvInner(ss,Ls,psi,chi,dee_v,uee_v,ueem_v,lee_v,leem_v);
  });

  MooeeInvTime+=usecond();

}


template<class Impl>
void
CayleyFermion5D<Impl>::MooeeInvDag (const FermionField &psi_i, FermionField &chi_i)
{
  chi_i.Checkerboard()=psi_i.Checkerboard();
  GridBase *grid=psi_i.Grid();
  
  auto psi = psi_i.View();
  auto chi = chi_i.View();
  auto lee_v  = &lee[0];
  auto leem_v = &leem[0];
  auto uee_v  = &uee[0];
  auto ueem_v = &ueem[0];
  auto dee_v  = &dee[0];
  int Ls=this->Ls;
  
  MooeeInvCalls++;
  MooeeInvTime-=usecond();
  uint64_t nloop = grid->oSites()/Ls;
  accelerator_for(sss,nloop,Simd::Nsimd(),{
    uint64_t ss=sss*Ls;
    MooeeInvDagInner(ss,Ls,psi,chi,dee_v,uee_v,ueem_v,lee_v,leem_v);
  });
  
  MooeeInvTime+=usecond();
  
}

template<class Impl>
void
CayleyFermion5D<Impl>::Meooe5DMooeeInv(const FermionField &in, FermionField &out) {
  
  this->tmp().Checkerboard() = in.Checkerboard();
  out.Checkerboard() = this->tmp().Checkerboard();
  GridBase *grid=in.Grid();
  
  auto in_v  = in.View();
  auto out_v = out.View();
  auto tmp_v = this->tmp().View();
  auto lee_v  = &lee[0];
  auto leem_v = &leem[0];
  auto uee_v  = &uee[0];
  auto ueem_v = &ueem[0];
  auto dee_v  = &dee[0];
  int Ls = this->Ls;
  
  Vector<Coeff_t> diag  = bs;
  Vector<Coeff_t> upper = cs;
  Vector<Coeff_t> lower = cs;
  upper[Ls-1] = -mass*upper[Ls-1];
  lower[0]    = -mass*lower[0];
  auto diag_v  = &diag[0];
  auto upper_v = &upper[0];
  auto lower_v = &lower[0];
  
  Meooe5DMooeeInvCalls++;
  Meooe5DMooeeInvTime-=usecond();
  
  uint64_t nloop = grid->oSites()/Ls;
  
  accelerator_for(sss,nloop,Simd::Nsimd(),{
    uint64_t ss=sss*Ls;
    MooeeInvInner(ss,Ls,in_v,tmp_v,dee_v,uee_v,ueem_v,lee_v,leem_v);
    M5DInner(ss,Ls,tmp_v,tmp_v,out_v,lower_v,diag_v,upper_v);
  });
  
  Meooe5DMooeeInvTime+=usecond();
  
}

template<class Impl>
void
CayleyFermion5D<Impl>::MooeeInvDagMeooeDag5D(const FermionField &in, FermionField &out) {
  
  this->tmp().Checkerboard() = in.Checkerboard();
  out.Checkerboard() = this->tmp().Checkerboard();
  GridBase *grid=in.Grid();
  
  auto in_v  = in.View();
  auto out_v = out.View();
  auto tmp_v = this->tmp().View();
  auto lee_v  = &lee[0];
  auto leem_v = &leem[0];
  auto uee_v  = &uee[0];
  auto ueem_v = &ueem[0];
  auto dee_v  = &dee[0];
  int Ls = this->Ls;
  
  Vector<Coeff_t> diag  = bs;
  Vector<Coeff_t> upper = cs;
  Vector<Coeff_t> lower = cs;
  for (int s=0;s<Ls;s++){
    if ( s== 0 ) {
      upper[s] = cs[s+1];
      lower[s] =-mass*cs[Ls-1];
    } else if ( s==(Ls-1) ) {
      upper[s] =-mass*cs[0];
      lower[s] = cs[s-1];
    } else {
      upper[s] = cs[s+1];
      lower[s] = cs[s-1];
    }
    upper[s] = conjugate(upper[s]);
    lower[s] = conjugate(lower[s]);
    diag[s]  = conjugate(diag[s]);
  }
  auto diag_v  = &diag[0];
  auto upper_v = &upper[0];
  auto lower_v = &lower[0];
  
  Meooe5DMooeeInvCalls++;
  Meooe5DMooeeInvTime-=usecond();
  
  uint64_t nloop = grid->oSites()/Ls;
  
  accelerator_for(sss,nloop,Simd::Nsimd(),{
    uint64_t ss=sss*Ls;
    M5DdagInner(ss,Ls,in_v,in_v,tmp_v,lower_v,diag_v,upper_v);
    MooeeInvDagInner(ss,Ls,tmp_v,out_v,dee_v,uee_v,ueem_v,lee_v,leem_v);
  });
  
  Meooe5DMooeeInvTime+=usecond();
  
}

NAMESPACE_END(Grid);
