
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the Maxwell equations

flux = FluxLaxFriedrichs(min_max_speed_naive)
equation = MaxwellEquations2D()
mesh = TreeMesh((-pi, -pi), (pi, pi), initial_refinement_level=4, n_cells_max=10^4)
solver = DGSEM(3, flux)
semi = SemidiscretizationHyperbolic(mesh, equation, Trixi.initial_condition_free_stream_conversion, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0,1*2.0*pi/299792458.0)
ode = semidiscretize(semi,tspan)
summary_callback = SummaryCallback()
callbacks = CallbackSet(summary_callback)

###############################################################################
# run the simulation
sol = solve(ode, SSPRK43(), save_everystep=false, callback=callbacks)