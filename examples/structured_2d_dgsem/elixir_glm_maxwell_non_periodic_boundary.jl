
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the Maxwell equations

equation = GLMMaxwellEquations2D(2.0)
boundary_conditions = (x_neg = Trixi.boundary_condition_irradiation,
                       x_pos = Trixi.boundary_condition_irradiation,
					   y_neg = Trixi.boundary_condition_perfect_conducting_wall,
					   y_pos = Trixi.boundary_condition_perfect_conducting_wall)
mesh = StructuredMesh((10, 10), (0.0, 0.0), (1.0, 1.0); periodicity = true)
solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)
semi = SemidiscretizationHyperbolic(mesh, equation, initial_condition_free_stream, solver
                                    )

###############################################################################
# ODE solvers, callbacks etc.


analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, save_analysis=true)
save_solution_callback = SaveSolutionCallback(interval = 100000, save_initial_solution=false, save_final_solution=true, output_directory="out")

cfl = 0.1
tspan = (0.0,1e-6)

ode = semidiscretize(semi,tspan)
summary_callback = SummaryCallback()
stepsize_callback = StepsizeCallback(cfl=cfl)
callbacks = CallbackSet(summary_callback,analysis_callback,stepsize_callback,save_solution_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
