/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Source file: ./lib/qcd/action/fermion/DomainWallEOFAFermioncache.cc

Copyright (C) 2017

Author: Peter Boyle <pabobyle@ph.ed.ac.uk>
Author: Peter Boyle <paboyle@ph.ed.ac.uk>
Author: Peter Boyle <peterboyle@Peters-MacBook-Pro-2.local>
Author: paboyle <paboyle@ph.ed.ac.uk>
Author: David Murphy <dmurphy@phys.columbia.edu>
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

#include <Grid/qcd/action/fermion/FermionCore.h>
#include <Grid/qcd/action/fermion/DomainWallEOFAFermion.h>

NAMESPACE_BEGIN(Grid);

// FIXME -- make a version of these routines with site loop outermost for cache reuse.
// Pminus fowards
// Pplus  backwards..
template<class Impl>
void DomainWallEOFAFermion<Impl>::M5D(const FermionField& psi_i, const FermionField& phi_i,FermionField& chi_i, 
				      Vector<Coeff_t>& lower, Vector<Coeff_t>& diag, Vector<Coeff_t>& upper)
{
  chi_i.Checkerboard() = psi_i.Checkerboard();
  int Ls = this->Ls;
  GridBase* grid = psi_i.Grid();
  assert(phi_i.Checkerboard() == psi_i.Checkerboard());
  // Flops = 6.0*(Nc*Ns) *Ls*vol
  this->M5Dcalls++;

  auto pdiag = &diag[0];
  auto pupper = &upper[0];
  auto plower = &lower[0];

  auto M5Dinner = [=] accelerator_only (uint64_t ss, auto psi, auto phi, auto chi) mutable {
    typedef decltype(coalescedRead(psi[0])) spinor;
    for(int s=0; s<Ls; s++){
      spinor tmp1, tmp2;
      uint64_t idx_u = ss+((s+1)%Ls);
      uint64_t idx_l = ss+((s+Ls-1)%Ls);
      spProj5m(tmp1, psi(idx_u));
      spProj5p(tmp2, psi(idx_l));
      coalescedWrite(chi[ss+s], pdiag[s]*phi(ss+s) + pupper[s]*tmp1 + plower[s]*tmp2);
    }
  };

  if( chi_i.list_is_unset() ) {
    auto psi = psi_i.View();
    auto phi = phi_i.View();
    auto chi = chi_i.View();
    uint64_t nloop = grid->oSites()/Ls;
    this->M5Dtime-=usecond();
    accelerator_for(sss,nloop,Simd::Nsimd(),{
        uint64_t ss=sss*Ls;
        M5Dinner(ss,psi,phi,chi);
      });
    this->M5Dtime+=usecond();
  } else if ( chi_i.get_list_size()!=0 ) {
    auto psi = psi_i.ViewList();
    auto phi = phi_i.ViewList();
    auto chi = chi_i.ViewList();
    uint64_t nloop = chi.size();
    accelerator_forNB(sss,nloop,Simd::Nsimd(),{
        uint64_t ss=chi.get_index(sss)*Ls;
        M5Dinner(ss,psi,phi,chi);
      });
  }

}

template<class Impl>
void DomainWallEOFAFermion<Impl>::M5Ddag(const FermionField& psi_i, const FermionField& phi_i, FermionField& chi_i, 
					 Vector<Coeff_t>& lower, Vector<Coeff_t>& diag, Vector<Coeff_t>& upper)
{
  chi_i.Checkerboard() = psi_i.Checkerboard();
  GridBase* grid = psi_i.Grid();
  int Ls = this->Ls;
  assert(phi_i.Checkerboard() == psi_i.Checkerboard());

  // Flops = 6.0*(Nc*Ns) *Ls*vol
  this->M5Dcalls++;

  auto pdiag = &diag[0];
  auto pupper = &upper[0];
  auto plower = &lower[0];

  auto M5Dinner = [=] accelerator_only (uint64_t ss, auto psi, auto phi, auto chi) mutable {
    typedef decltype(coalescedRead(psi[0])) spinor;
    for(int s=0; s<Ls; s++){
      spinor tmp1, tmp2;
      uint64_t idx_u = ss+((s+1)%Ls);
      uint64_t idx_l = ss+((s+Ls-1)%Ls);
      spProj5p(tmp1, psi(idx_u));
      spProj5m(tmp2, psi(idx_l));
      coalescedWrite(chi[ss+s], pdiag[s]*phi(ss+s) + pupper[s]*tmp1 + plower[s]*tmp2);
    }
  };

  if( chi_i.list_is_unset() ) {
    auto psi = psi_i.View();
    auto phi = phi_i.View();
    auto chi = chi_i.View();
    uint64_t nloop = grid->oSites()/Ls;
    this->M5Dtime-=usecond();
    accelerator_for(sss,nloop,Simd::Nsimd(),{
        uint64_t ss=sss*Ls;
        M5Dinner(ss,psi,phi,chi);
      });
    this->M5Dtime+=usecond();
  } else if ( chi_i.get_list_size()!=0 ) {
    auto psi = psi_i.ViewList();
    auto phi = phi_i.ViewList();
    auto chi = chi_i.ViewList();
    uint64_t nloop = chi.size();
    accelerator_forNB(sss,nloop,Simd::Nsimd(),{
        uint64_t ss=chi.get_index(sss)*Ls;
        M5Dinner(ss,psi,phi,chi);
      });
  }
}

template<class Impl>
void DomainWallEOFAFermion<Impl>::MooeeInv(const FermionField& psi_i, FermionField& chi_i)
{
  chi_i.Checkerboard() = psi_i.Checkerboard();
  GridBase* grid = psi_i.Grid();
  int Ls = this->Ls;

  auto plee  = & this->lee[0];
  auto pdee  = & this->dee[0];
  auto puee  = & this->uee[0];

  auto pleem = & this->leem[0];
  auto pueem = & this->ueem[0];

  this->MooeeInvCalls++;

  auto MooeeInvinner = [=] accelerator_only (uint64_t ss, auto psi, auto chi) mutable {
    typedef decltype(coalescedRead(psi[0])) spinor;
    spinor tmp1,tmp2;

    // flops = 12*2*Ls + 12*2*Ls + 3*12*Ls + 12*2*Ls  = 12*Ls * (9) = 108*Ls flops
    // Apply (L^{\prime})^{-1}
    coalescedWrite(chi[ss],psi(ss)); // chi[0]=psi[0]
    for(int s=1; s<Ls; s++){
      spProj5p(tmp1, chi(ss+s-1));
      coalescedWrite(chi[ss+s], psi(ss+s) - plee[s-1]*tmp1);
    }

    // L_m^{-1}
    for(int s=0; s<Ls-1; s++){ // Chi[ee] = 1 - sum[s<Ls-1] -leem[s]P_- chi
      spProj5m(tmp1, chi(ss+s));
      coalescedWrite(chi[ss+Ls-1], chi(ss+Ls-1) - pleem[s]*tmp1);
    }

    // U_m^{-1} D^{-1}
    for(int s=0; s<Ls-1; s++){ // Chi[s] + 1/d chi[s]
      spProj5p(tmp1, chi(ss+Ls-1));
      coalescedWrite(chi[ss+s], (1.0/pdee[s])*chi(ss+s) - (pueem[s]/pdee[Ls])*tmp1);
    }
    spProj5m(tmp2, chi(ss+Ls-1));
    coalescedWrite(chi[ss+Ls-1],(1.0/pdee[Ls])*tmp1 + (1.0/pdee[Ls-1])*tmp2);

    // Apply U^{-1}
    for(int s=Ls-2; s>=0; s--){
      spProj5m(tmp1, chi(ss+s+1));
      coalescedWrite(chi[ss+s], chi(ss+s) - puee[s]*tmp1);
    }
  };

  if( chi_i.list_is_unset() ) {
    auto psi = psi_i.View();
    auto chi = chi_i.View();
    uint64_t nloop = grid->oSites()/Ls;
    this->MooeeInvTime-=usecond();
    accelerator_for(sss,nloop,Simd::Nsimd(),{
        uint64_t ss=sss*Ls;
        MooeeInvinner(ss,psi,chi);
      });
    this->MooeeInvTime+=usecond();
  } else if ( chi_i.get_list_size()!=0 ) {
    auto psi = psi_i.ViewList();
    auto chi = chi_i.ViewList();
    uint64_t nloop = chi.size();
    accelerator_forNB(sss,nloop,Simd::Nsimd(),{
        uint64_t ss=chi.get_index(sss)*Ls;
        MooeeInvinner(ss,psi,chi);
      });
  }
}

template<class Impl>
void DomainWallEOFAFermion<Impl>::MooeeInvDag(const FermionField& psi_i, FermionField& chi_i)
{
  chi_i.Checkerboard() = psi_i.Checkerboard();
  GridBase* grid = psi_i.Grid();
  int Ls = this->Ls;
  assert(psi_i.Checkerboard() == psi_i.Checkerboard());

  Vector<Coeff_t> ueec(Ls);
  Vector<Coeff_t> deec(Ls+1);
  Vector<Coeff_t> leec(Ls);
  Vector<Coeff_t> ueemc(Ls);
  Vector<Coeff_t> leemc(Ls);

  for(int s=0; s<ueec.size(); s++){
    ueec[s]  = conjugate(this->uee[s]);
    deec[s]  = conjugate(this->dee[s]);
    leec[s]  = conjugate(this->lee[s]);
    ueemc[s] = conjugate(this->ueem[s]);
    leemc[s] = conjugate(this->leem[s]);
  }
  deec[Ls] = conjugate(this->dee[Ls]);

  auto pueec = &ueec[0];
  auto pdeec = &deec[0];
  auto pleec = &leec[0];
  auto pueemc = &ueemc[0];
  auto pleemc = &leemc[0];

  this->MooeeInvCalls++;

  auto MooeeInvinner = [=] accelerator_only (uint64_t ss, auto psi, auto chi) mutable {
    typedef decltype(coalescedRead(psi[0])) spinor;
    spinor tmp1,tmp2;

    // Apply (U^{\prime})^{-dagger}
    coalescedWrite(chi[ss], psi(ss));
    for(int s=1; s<Ls; s++){
      spProj5m(tmp1, chi(ss+s-1));
      coalescedWrite(chi[ss+s], psi(ss+s) - pueec[s-1]*tmp1);
    }

    // U_m^{-\dagger}
    for(int s=0; s<Ls-1; s++){
      spProj5p(tmp1, chi(ss+s));
      coalescedWrite(chi[ss+Ls-1], chi(ss+Ls-1) - pueemc[s]*tmp1);
    }

    // L_m^{-\dagger} D^{-dagger}
    for(int s=0; s<Ls-1; s++){
      spProj5m(tmp1, chi(ss+Ls-1));
      coalescedWrite(chi[ss+s] ,(1.0/pdeec[s])*chi(ss+s) - (pleemc[s]/pdeec[Ls-1])*tmp1);
    }
    spProj5p(tmp2, chi(ss+Ls-1));
    coalescedWrite(chi[ss+Ls-1], (1.0/pdeec[Ls-1])*tmp1 + (1.0/pdeec[Ls])*tmp2);

    // Apply L^{-dagger}
    for(int s=Ls-2; s>=0; s--){
      spProj5p(tmp1, chi(ss+s+1));
      coalescedWrite(chi[ss+s],chi(ss+s) - pleec[s]*tmp1);
    }
  };

  if( chi_i.list_is_unset() ) {
    auto psi = psi_i.View();
    auto chi = chi_i.View();
    uint64_t nloop = grid->oSites()/Ls;
    this->MooeeInvTime-=usecond();
    accelerator_for(sss,nloop,Simd::Nsimd(),{
        uint64_t ss=sss*Ls;
        MooeeInvinner(ss,psi,chi);
      });
    this->MooeeInvTime+=usecond();
  } else if ( chi_i.get_list_size()!=0 ) {
    auto psi = psi_i.ViewList();
    auto chi = chi_i.ViewList();
    uint64_t nloop = chi.size();
    accelerator_forNB(sss,nloop,Simd::Nsimd(),{
        uint64_t ss=chi.get_index(sss)*Ls;
        MooeeInvinner(ss,psi,chi);
      });
  }
}

NAMESPACE_END(Grid);
