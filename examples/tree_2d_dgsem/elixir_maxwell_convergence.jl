
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the Maxwell equations

flux = FluxLaxFriedrichs(min_max_speed_naive)
equation = MaxwellEquations2D()
mesh = TreeMesh((0.0, 0.0), (1.0, 1.0), initial_refinement_level=4, n_cells_max=10^4)
solver = DGSEM(3, flux)
semi = SemidiscretizationHyperbolic(mesh, equation, Trixi.initial_condition_convergence, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0,1e-6)
ode = semidiscretize(semi,tspan)
summary_callback = SummaryCallback()
callbacks = CallbackSet(summary_callback)

###############################################################################
# run the simulation

sol = solve(ode, SSPRK43(), save_everystep=false, callback=callbacks)