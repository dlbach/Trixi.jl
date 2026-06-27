using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the Maxwell equations

function initial_condition_orszag_tang(x, t, equations::Trixi.GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D)
    gamma_i = equations.gammas[2]
    gamma_e = equations.gammas[1]
    R_i = equations.gas_constants[2]
    R_e = equations.gas_constants[1]
    L = 4 * pi
    delta_B = 0.2
    delta_u = 0.2
    p_i = R_i * 0.05
    p_e = R_e * 0.05 * 0.04
    B_0 = 1.0 #sqrt(2 * p_i)

    rho_i = 1.0
    rho_e = 0.04 * (1 + 2 * pi * delta_u * B_0 * (cospi(2 * x[1] / L) + cospi(2 * x[2] / L)) * equations.permittivity / (equations.c_e * L))
    rho_v1_i = -delta_u * sinpi(2 * x[2] / L) * rho_i
    rho_v2_i = delta_u * sinpi(2 * x[1] / L) * rho_i
    rho_v3_i = 0.0
    rho_v1_e = -delta_u * sinpi(2 * x[2] / L) * rho_e
    rho_v2_e = delta_u * sinpi(2 * x[1] / L) * rho_e
    rho_v3_e = -2 * pi * delta_B * (2 * cospi(4 * x[1] / L) + cospi(2 * x[2] / L)) * rho_e / L
    rho_s_i = rho_i * log(p_i / rho_i^gamma_i)
    rho_s_e = rho_e * log(p_e / rho_e^gamma_e)

    E1 = -delta_u * B_0 * sinpi(2 * x[1] / L) 
    E2 = -delta_u * B_0 * sinpi(2 * x[2] / L)
    E3 = 0.0
    B1 = -delta_B * B_0 * sinpi(2 * x[2] / L)
    B2 = delta_B * B_0 * sinpi(4 * x[1] / L)
    B3 = B_0
    psi_E = 0.0
    psi_B = 0.0

    return SVector(rho_e, rho_v1_e, rho_v2_e, rho_v3_e, rho_s_e, rho_i, rho_v1_i, rho_v2_i, rho_v3_i, rho_s_i, E1, E2, E3, B1, B2, B3, psi_E, psi_B)
end

function Trixi.get_node_variable(::Val{:current_density_z}, u, mesh, equations, dg, cache)
    n_nodes = nnodes(dg)
    n_elements = nelements(dg, cache)
    # By definition, the variable must be provided at every node of every element!
    # Otherwise, the `SaveSolutionCallback` will crash.
    current_density_array = zeros(eltype(cache.elements),
                            n_nodes, n_nodes, # equivalent: `ntuple(_ -> n_nodes, ndims(mesh))...,`
                            n_elements)

    # We can accelerate the computation by thread-parallelizing the loop over elements
    # by using the `@threaded` macro.
    Trixi.@threaded for element in eachelement(dg, cache)
        for j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, element)

            current_density_nodal = Trixi.current_density(u_node, equations)
            current_density_array[i, j, element] = current_density_nodal[3]
        end
    end

    return current_density_array
end


speed_of_light = 10.0
permittivity = inv(speed_of_light^2)
volume_flux = Trixi.flux_energy_central
surface_flux = Trixi.flux_lax_friedrichs
equation = Trixi.GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D((1.6, 1.6), (25.0, 1.0), (-25.0, 1.0), speed_of_light, permittivity)
coordinates_min = (0.0, 0.0)
coordinates_max = (4*pi, 4*pi)

basis = LobattoLegendreBasis(2)
mesh = TreeMesh(coordinates_min, coordinates_max, periodicity = true, initial_refinement_level = 7, n_cells_max = 10^8)

indicator_sc = IndicatorHennemannGassner(equation, basis,
                                         alpha_max = 0.5,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = Trixi.density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)

solver = DGSEM(basis, surface_flux, volume_integral) #VolumeIntegralFluxDifferencing(volume_flux))
semi = SemidiscretizationHyperbolic(mesh, equation,
                                    initial_condition_orszag_tang, solver, source_terms = Trixi.source_term_lorentz_corrected_2)

###############################################################################
# ODE solvers, callbacks etc.
amr_indicator = IndicatorHennemannGassner(semi,
                                          alpha_max = 0.5,
                                          alpha_min = 0.001,
                                          alpha_smooth = true,
                                          variable = Trixi.density_pressure)
amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level = 7,
                                      max_level = 10, max_threshold = 0.002)
amr_callback = AMRCallback(semi, amr_controller,
                           interval = 6,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true)
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     save_analysis = true,
                                     output_directory = "out"
                                     )
save_solution = SaveSolutionCallback(dt = 0.5,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim,
                                     output_directory = "out",
                                     extra_node_variables = (:current_density_z,)
                                    )
cfl = 1.0
tspan = (0.0, 30.0)

ode = semidiscretize(semi, tspan)
summary_callback = SummaryCallback()
stepsize_callback = StepsizeCallback(cfl = cfl)
callbacks = CallbackSet(summary_callback, analysis_callback, save_solution, amr_callback)
###############################################################################
# run the simulation
#=
sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
=#

# Use a Runge-Kutta method with automatic (error based) time step size control
# Enable threading of the RK method for better performance on multiple threads

sol = solve(ode, RDPK3SpFSAL49(thread = Trixi.True());
            abstol = 1.0e-8, reltol = 1.0e-8,
            ode_default_options()..., callback = callbacks, maxiters = 100000000);
