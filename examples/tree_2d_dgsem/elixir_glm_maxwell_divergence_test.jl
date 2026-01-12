using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the Maxwell equations

function initial_condition_divergence_test(x, t, equations::GLMMaxwellEquations2D)
    return SVector(0.0f0, 0.0f0, 0.0f0, 0.0f0)
end

function source_term_function(u, x, t, equations::GLMMaxwellEquations2D)
    return SVector(0.0, 0.0, 0.0, x[1])
end

equation = GLMMaxwellEquations2D(299_792_458.0, 1.0)
mesh = TreeMesh((-1.0, -1.0), (1.0, 1.0), initial_refinement_level = 2, n_cells_max = 10^4)
solver = DGSEM(3, Trixi.flux_upwind)
semi = SemidiscretizationHyperbolic(mesh, equation,
                                    initial_condition_divergence_test, solver,
                                    source_terms = source_term_function)

###############################################################################
# ODE solvers, callbacks etc.

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     save_analysis = true)

cfl = 1.0
tspan = (0.0, 1e-8)

ode = semidiscretize(semi, tspan)
summary_callback = SummaryCallback()
stepsize_callback = StepsizeCallback(cfl = cfl)
callbacks = CallbackSet(summary_callback, analysis_callback, stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
