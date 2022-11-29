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
	return SVector(10.0,10.0,10.0)
end


min_max_speed_naive(u_ll, u_rr, orientation, ::MaxwellEquations2D) = 299792458.0

flux = FluxLaxFriedrichs(min_max_speed_naive)
equation = MaxwellEquations2D()
mesh = TreeMesh((0.0, 0.0), (1.0, 1.0), initial_refinement_level=4, n_cells_max=10^4)
solver = DGSEM(3, flux)
semi = SemidiscretizationHyperbolic(mesh, equation, initial_condition_test, solver)

tspan = (0.0,1e-6)
ode = semidiscretize(semi,tspan)
summary_callback = SummaryCallback()
callbacks = CallbackSet(summary_callback)

odestart = semidiscretize(semi,(0.0,0.0))

start = solve(odestart, SSPRK43(), save_everystep=false, callback=callbacks)
sol = solve(ode, SSPRK43(), save_everystep=false, callback=callbacks)