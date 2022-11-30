# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

struct MaxwellEquations2D <: AbstractGLMMaxwellEquations{2,3}
end


function flux(u, orientation::Integer, equation::MaxwellEquations2D)
  c_sqr = 299792458.0^2
  if orientation == 1
    f1 = 0.0
    f2 = c_sqr*u[3]
    f3 = u[2]
  else
    f1 = -c_sqr*u[3]
    f2 = 0.0
    f3 = -u[1]
  end
  return Trixi.SVector(f1,f2,f3)
end

Trixi.varnames(::typeof(cons2cons), ::MaxwellEquations2D) = ("E1","E2","B")

function initial_condition_test(x, t, equations::MaxwellEquations2D)
  c = 299792458.0
  c_sqr = 299792458.0^2
  ω = sqrt(2)*pi*c
  k_orth = pi
  k_par = pi
  e1 = abs(-(k_orth/k_par)*sin(k_orth*x[2])*cos(k_par*x[1]))
  e2 = abs(cos(k_orth*x[2])*sin(k_par*x[1]))
  b = abs((ω/(k_par*c_sqr))*cos(k_orth*x[2])*sin(k_par*x[1]))
	
  return SVector(e1,e2,b)
end

function initial_condition_free_stream_conversion(x, t, equations::MaxwellEquations2D)
  c = 299792458.0
  return SVector(10.0,10.0,10.0/c)
end

function initial_condition_convergence(x, t, equations::MaxwellEquations2D)
  c = 299792458.0
  e1 = sin(x[2])
  e2 = -sin(x[1])
  b = -(sin(x[1])+sin(x[2]))/c
  
  return SVector(e1,e2,b)
end

min_max_speed_naive(u_ll, u_rr, orientation, ::MaxwellEquations2D) = 299792458.0

end # @muladd
