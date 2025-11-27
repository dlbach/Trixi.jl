
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the Maxwell equations

f1(xi) = SVector(xi/sqrt(2), -sqrt(1.0 - 0.5 * xi^2))
f2(xi) = SVector(xi/sqrt(2), sqrt(1.0 - 0.5 * xi^2))
f3(eta) = SVector(-sqrt(1.0 - 0.5 * eta^2), eta/sqrt(2))
f4(eta) = SVector(sqrt(1.0 - 0.5 * eta^2), eta/sqrt(2))
"""
f1(xi) = SVector(0.2*cos((xi+1.0)*pi),0.2*sin((xi+1.0)*pi))
f2(xi) = SVector(cos((xi+1.0)*pi),sin((xi+1.0)*pi))
f3(eta) = SVector(0.4*(eta+1.0)+0.2,0.0)
f4(eta) = SVector(0.4*(eta+1.0)+0.2,0.0)
"""

cells_per_dimension = (5, 5)

function boundary_condition_irradiation(u_inner, orientation_or_normal_direction, direction,
                                        x, t,
                                        surface_flux_function,
                                        equations::GLMMaxwellEquations2D)
    if iseven(direction)
        return surface_flux_function(u_inner,
                                     SVector(2.0*x[1]-u_inner[1], 2.0*x[2]-u_inner[2],
                                             u_inner[3], -u_inner[4]),
                                     orientation_or_normal_direction, equations)
    else
        return surface_flux_function(SVector(2.0*x[1]-u_inner[1], 2.0*x[2]-u_inner[2],
                                             u_inner[3], -u_inner[4]), u_inner,
                                     orientation_or_normal_direction, equations)
    end
end

function source_term_function(u, x, t, equations::GLMMaxwellEquations2D)
    return SVector(0.0, 0.0, 0.0, equations.c_h^2 * 2.0)
end

boundary_dir(x, t, equations::GLMMaxwellEquations2D) = SVector(x[1], x[2], 0.0, 0.0)

function initial_condition_zero(x, t, equations::GLMMaxwellEquations2D)
    if t > 0.0
        return SVector(x[1], x[2], 0.0, 0.0)
    else
        return SVector(0.0, 0.0, 0.0, 0.0)
    end
end

equation = GLMMaxwellEquations2D(299_792_458.0, 1000.0)
boundary_conditions = Trixi.boundary_condition_perfect_conducting_wall
"""
boundary_conditions = (y_neg = Trixi.boundary_condition_periodic,
                       y_pos = Trixi.boundary_condition_periodic,
					   x_neg = Trixi.boundary_condition_perfect_conducting_wall,
					   x_pos = Trixi.boundary_condition_perfect_conducting_wall)"""
mesh = StructuredMesh(cells_per_dimension, (f1, f2, f3, f4), periodicity = false)
solver = DGSEM(polydeg = 3, surface_flux = Trixi.flux_upwind)
semi = SemidiscretizationHyperbolic(mesh, equation, initial_condition_zero, solver,
                                    boundary_conditions = boundary_conditions,
                                    source_terms = source_term_function)

###############################################################################
# ODE solvers, callbacks etc.

analysis_interval = 100000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     save_analysis = true)
save_solution_callback = SaveSolutionCallback(interval = 100000,
                                              save_initial_solution = false,
                                              save_final_solution = true,
                                              output_directory = "out")

cfl = 0.3
tspan = (0.0, 1e-7)

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
