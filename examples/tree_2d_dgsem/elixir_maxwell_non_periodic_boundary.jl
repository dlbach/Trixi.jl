using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the Maxwell equations

equation = MaxwellEquations2D()
boundary_conditions = (x_neg = Trixi.boundary_condition_irradiation,
                       x_pos = Trixi.boundary_condition_truncation,
                       y_neg = Trixi.boundary_condition_perfect_conducting_wall,
                       y_pos = Trixi.boundary_condition_perfect_conducting_wall)
mesh = TreeMesh((0.0, 0.0), (1.0, 1.0), initial_refinement_level = 2, n_cells_max = 10^4,
                periodicity = false)
solver = DGSEM(3, Trixi.flux_upwind)
semi = SemidiscretizationHyperbolic(mesh, equation, Trixi.initial_condition_test, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     save_analysis = true)

cfl = 1.0
tspan = (0.0, 1e-6)

ode = semidiscretize(semi, tspan)
summary_callback = SummaryCallback()
stepsize_callback = StepsizeCallback(cfl = cfl)
callbacks = CallbackSet(summary_callback, analysis_callback, stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
