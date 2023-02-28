
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the Maxwell equations

equation = GLMMaxwellEquations2D(2.0)
boundary_conditions = (x_neg = Trixi.boundary_condition_irradiation,
                       x_pos = Trixi.boundary_condition_irradiation,
					   y_neg = Trixi.boundary_condition_irradiation,
					   y_pos = Trixi.boundary_condition_irradiation)
mesh = TreeMesh((0.0, 0.0), (1.0, 1.0), initial_refinement_level=2, n_cells_max=10^4, periodicity = false)
solver = DGSEM(polydeg = 3, surface_flux = Trixi.flux_upwind)
semi = SemidiscretizationHyperbolic(mesh, equation, Trixi.initial_condition_test, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.


analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, save_analysis=true)
save_solution_callback = SaveSolutionCallback(interval = 100000, save_initial_solution=false, save_final_solution=true, output_directory="out")

cfl = 1.0
tspan = (0.0,1e-7)

ode = semidiscretize(semi,tspan)
summary_callback = SummaryCallback()
stepsize_callback = StepsizeCallback(cfl=cfl)
callbacks = CallbackSet(summary_callback,analysis_callback,stepsize_callback,save_solution_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);