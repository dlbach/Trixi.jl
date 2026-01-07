using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the Maxwell equations
function Trixi.boundary_condition_irradiation(u_inner, orientation_or_normal_direction, direction,
                                        x, t,
                                        surface_flux_function,
                                        equations::GLMMaxwellEquations2D)
    c = equations.speed_of_light
    factor = 2 * equations.c_h / c
    k_orth = pi
    k_par = pi
    omega = sqrt(k_orth^2 + k_par^2)*c
    e1 = -(k_orth/k_par)*sin(k_orth*x[2])*cos(k_par*x[1]-omega*t)
    e2 = cos(k_orth*x[2])*sin(k_par*x[1]-omega*t)
    if iseven(direction)
        return surface_flux_function(u_inner,
                                     SVector(2 * e1 - u_inner[1], 2 * e2 - u_inner[2],
                                             u_inner[3], u_inner[4] + factor * (e1 - u_inner[1])),
                                     orientation_or_normal_direction, equations)
    else
        return surface_flux_function(SVector(2 * e1 - u_inner[1], 2 * e2 - u_inner[2],
                                             u_inner[3], u_inner[4] - factor * (e1 - u_inner[1])), u_inner,
                                     orientation_or_normal_direction, equations)
    end
end

function Trixi.boundary_condition_truncation(u_inner, orientation_or_normal_direction, direction,
                                        x, t,
                                        surface_flux_function,
                                        equations::GLMMaxwellEquations2D)
    if iseven(direction)
        return surface_flux_function(u_inner,
                                    SVector(0, 0, 0, 0),
                                    orientation_or_normal_direction, equations)
    else
        return surface_flux_function(SVector(0, 0, 0, 0), u_inner,
                                    orientation_or_normal_direction, equations)
    end
end

equation = GLMMaxwellEquations2D(299_792_458.0, 0.00001)
boundary_conditions = (x_neg = Trixi.boundary_condition_irradiation,
                       x_pos = Trixi.boundary_condition_truncation,
                       y_neg = Trixi.boundary_condition_perfect_conducting_wall,
                       y_pos = Trixi.boundary_condition_perfect_conducting_wall)
mesh = TreeMesh((0.0, 0.0), (1.0, 1.0), initial_refinement_level = 2, n_cells_max = 10^4,
                periodicity = false)
solver = DGSEM(polydeg = 3, surface_flux = Trixi.flux_upwind)
semi = SemidiscretizationHyperbolic(mesh, equation, Trixi.initial_condition_test, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     save_analysis = true)
save_solution_callback = SaveSolutionCallback(interval = 100000,
                                              save_initial_solution = false,
                                              save_final_solution = true,
                                              output_directory = "out")

cfl = 0.5
tspan = (0.0, 3e-7)

ode = semidiscretize(semi, tspan)
summary_callback = SummaryCallback()
stepsize_callback = StepsizeCallback(cfl = cfl)
callbacks = CallbackSet(summary_callback, analysis_callback, stepsize_callback,
                        save_solution_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
