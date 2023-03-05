
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the Maxwell equations

###############################################################################
# semidiscretization of the Maxwell equations
function boundary_condition_irradiation(u_inner, orientation_or_normal_direction, direction, x, t,
  surface_flux_function, equations::MaxwellEquations2D)
  if iseven(direction)
    return surface_flux_function(u_inner, SVector(2.0*x[1]-u_inner[1], 2.0*x[2]-u_inner[2], u_inner[3]), orientation_or_normal_direction, equations)
  else
    return surface_flux_function(SVector(2.0*x[1]-u_inner[1], 2.0*x[2]-u_inner[2], u_inner[3]), u_inner, orientation_or_normal_direction, equations)
  end
end


function boundary_value_function(x, t, equations::MaxwellEquations2D)
  return SVector(2.0*x[1], 0.0, 0.0)
end


function source_term_function(u, x, t, equations::MaxwellEquations2D)
  return SVector(0.0, 0.0, 0.0)
end

function initial_condition_zero(x, t, equations::MaxwellEquations2D)
  if t > 0.0
    return SVector(2.0*x[1], 0.0 , 0.0)
  else 
    return SVector(0.0, 0.0, 0.0)
  end
end

equation = MaxwellEquations2D()
boundary_conditions = (x_neg = Trixi.boundary_condition_perfect_conducting_wall,
                       x_pos = Trixi.boundary_condition_perfect_conducting_wall,
					   y_neg = BoundaryConditionDirichlet(boundary_value_function),
					   y_pos = BoundaryConditionDirichlet(boundary_value_function))
mesh = TreeMesh((-1.0, -1.0), (1.0, 1.0), initial_refinement_level=2, n_cells_max=10^4, periodicity = false)
solver = DGSEM(polydeg = 3, surface_flux = Trixi.flux_upwind)
semi = SemidiscretizationHyperbolic(mesh, equation, initial_condition_zero, solver,
                                    boundary_conditions = boundary_conditions, source_terms = source_term_function)

###############################################################################
# ODE solvers, callbacks etc.


analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, save_analysis=true)
save_solution_callback = SaveSolutionCallback(interval = 100, save_initial_solution=false, save_final_solution=true, output_directory="out")

cfl = 1.0
tspan = (0.0,1e-5)

ode = semidiscretize(semi,tspan)
summary_callback = SummaryCallback()
stepsize_callback = StepsizeCallback(cfl=cfl)
callbacks = CallbackSet(summary_callback,analysis_callback,stepsize_callback,save_solution_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
