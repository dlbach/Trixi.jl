using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the Maxwell equations

function initial_condition_linear_electric(x, t, equations::GLMMaxwellEquations2D)
    return SVector(x[1]*x[2], x[1]*x[2], 0.0, 0.0)
end
coordinates_min = (0.0, 0.0)
coordinates_max = (1.0, 1.0)
refinement_patches = ((type = "box", coordinates_min = (0.0, 0.0),
                       coordinates_max = (0.5, 0.5)), )

equation = GLMMaxwellEquations2D(299_792_458.0, 1.0)
mesh = TreeMesh(coordinates_min, coordinates_max, initial_refinement_level = 1, n_cells_max = 10^4, refinement_patches = refinement_patches)
solver = DGSEM(polydeg = 2, surface_flux = Trixi.flux_upwind)
semi = SemidiscretizationHyperbolic(mesh, equation, initial_condition_linear_electric, solver)

###############################################################################
# ODE solvers, callbacks etc.

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     save_analysis = true)
save_solution_callback = SaveSolutionCallback(interval = 100, save_initial_solution = false)

cfl = 0.9
tspan = (0.0, 0.0)

ode = semidiscretize(semi, tspan)
summary_callback = SummaryCallback()
stepsize_callback = StepsizeCallback(cfl = cfl)
callbacks = CallbackSet(summary_callback, analysis_callback, stepsize_callback,
                        save_solution_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks, maxiters = Inf);
