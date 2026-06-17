using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the Maxwell equations

function initial_condition_convergence_2(x, t, equations::GlmMaxwellEquations2D)
    u5 = sin(2*pi*x[1])*cos(2*pi*x[2])*sin(2*pi*t)
    u6 = -cos(2*pi*x[1])*sin(2*pi*x[2])*sin(2*pi*t)
    u7 = cos(2*pi*x[1])*sin(2*pi*t)#/equations.speed_of_light
    u8 = sin(2*pi*x[1])*cos(2*pi*t)#/equations.speed_of_light
    return SVector(u5, u6, u7, u8)
end

function source_terms_convergence(u, x, t, equations::GlmMaxwellEquations2D)
    s5 = 2*pi*(sin(2*pi*x[1])*cos(2*pi*x[2])*cos(2*pi*t) + cos(2*pi*x[1])*cos(2*pi*t)*equations.speed_of_light^2)
    s6 = -2*pi*(cos(2*pi*x[1])*sin(2*pi*x[2])*cos(2*pi*t) + sin(2*pi*x[1])*sin(2*pi*t)*equations.speed_of_light^2)
    s7 = 2*pi*cos(2*pi*x[1])*cos(2*pi*t) + 4*pi*sin(2*pi*x[1])*sin(2*pi*x[2])*sin(2*pi*t)
    #s7 = 2*pi*cos(2*pi*x[1])*cos(2*pi*t)/equations.speed_of_light + 4*pi*sin(2*pi*x[1])*sin(2*pi*x[2])*sin(2*pi*t)
    s8 = -2*pi*sin(2*pi*x[1])*sin(2*pi*t)#/equations.speed_of_light
    return SVector(s5, s6, s7, s8)
end


equation = GlmMaxwellEquations2D(1e0, 1e1)
mesh = TreeMesh((-1.0, -1.0), (1.0, 1.0), initial_refinement_level = 2, n_cells_max = 10^7)
solver = DGSEM(3, Trixi.flux_lax_friedrichs)
semi = SemidiscretizationHyperbolic(mesh, equation,
                                    initial_condition_convergence_2, solver, source_terms = source_terms_convergence)

###############################################################################
# ODE solvers, callbacks etc.

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     save_analysis = true)

cfl = 0.005
tspan = (0.0, 1e-4)

ode = semidiscretize(semi, tspan)
summary_callback = SummaryCallback()
stepsize_callback = StepsizeCallback(cfl = cfl)
callbacks = CallbackSet(summary_callback, analysis_callback, stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
