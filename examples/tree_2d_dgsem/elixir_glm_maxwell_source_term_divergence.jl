using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the Maxwell equations

function source_term_function(u, x, t, equations::GLMMaxwellEquations2D)
    perm_inv = 1.129409067e11
    omega = 1e-9
    ct = cos(omega*t)
    s1 = -perm_inv * ((ct - 1.0) * (pi * cos(pi*x[1]) + pi^2 * x[1] * sin(pi*x[2])) -
          x[1] * ct * sin(pi*x[2]))
    s2 = -perm_inv * ((ct - 1.0) * (pi * cos(pi*x[2]) + pi^2 * x[2] * sin(pi*x[1])) -
          x[2] * ct * sin(pi*x[1]))
    s4 = equations.c_h^2 * perm_inv * sin(omega*t) * (sin(pi*x[1]) + sin(pi*x[2]))
    return SVector(s1, s2, 0.0, s4)
end

function initial_condition_zero(x, t, equations::GLMMaxwellEquations2D)
    return SVector(0.0, 0.0, 0.0, 0.0)
end

equation = GLMMaxwellEquations2D(299_792_458.0, 100.0)
mesh = TreeMesh((0.0, 0.0), (1.0, 1.0), initial_refinement_level = 2, n_cells_max = 10^4)
solver = DGSEM(polydeg = 3, surface_flux = Trixi.flux_upwind)
semi = SemidiscretizationHyperbolic(mesh, equation, initial_condition_zero, solver
                                    source_terms = source_term_function)

###############################################################################
# ODE solvers, callbacks etc.

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     save_analysis = true)
save_solution_callback = SaveSolutionCallback(interval = 100, save_initial_solution = false,
                                              save_final_solution = true,
                                              output_directory = "out")

cfl = 0.5e-1
tspan = (0.0, 5e-9)

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
