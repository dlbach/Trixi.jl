using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the Maxwell equations

function initial_condition_orszag_tang(x, t, equations::Trixi.GlmMultiFluid5MomentPlasmaEquationsEntropy3D)
    gamma_i = equations.gammas[2]
    gamma_e = equations.gammas[1]
    R_i = equations.gas_constants[2]
    R_e = equations.gas_constants[1]
    delta_B = 1.0
    delta_u = 1.0
    p_i = inv(gamma_i) #R_i * 0.05
    p_e = inv(gamma_e) #R_e * 0.05 * 0.04
    B_0 = 1.0 #sqrt(2 * p_i)

    rho_i = 1.0
    rho_e = 0.04 # * (1 + 2 * pi * delta_u * B_0 * (cospi(2 * x[1]) + cospi(2 * x[2])) * equations.permittivity / equations.c_e)
    rho_v1_i = -delta_u * sinpi(2 * x[2]) * rho_i
    rho_v2_i = delta_u * sinpi(2 * x[1]) * rho_i
    rho_v3_i = 0.0
    rho_v1_e = -delta_u * sinpi(2 * x[2]) * rho_e
    rho_v2_e = delta_u * sinpi(2 * x[1]) * rho_e
    rho_v3_e = 0.0 #-2 * pi * delta_u * (2 * cospi(4 * x[1]) + cospi(2 * x[2])) * rho_e
    rho_s_i = rho_i * log(p_i / rho_i^gamma_i)
    rho_s_e = rho_e * log(p_e / rho_e^gamma_e)

    E1 = 0.0  #-delta_u * B_0 * sinpi(2 * x[1]) 
    E2 = 0.0 #-delta_u * B_0 * sinpi(2 * x[2])
    E3 = (-delta_u * sinpi(2 * x[2]) * delta_B * B_0 * sinpi(4 * x[1]) / gamma_i + delta_u * sinpi(2 * x[2]) * delta_B * B_0 * sinpi(4 * x[1]) / gamma_i )
    B1 = -delta_B * B_0 * sinpi(2 * x[2]) / gamma_i
    B2 = delta_B * B_0 * sinpi(4 * x[1]) / gamma_i
    B3 = 0.0 #B_0
    psi_E = 0.0
    psi_B = 0.0

    return SVector(rho_e, rho_v1_e, rho_v2_e, rho_v3_e, rho_s_e, rho_i, rho_v1_i, rho_v2_i, rho_v3_i, rho_s_i, E1, E2, E3, B1, B2, B3, psi_E, psi_B)
end

volume_flux = Trixi.flux_energy_central
surface_flux = Trixi.flux_energy_upwind
speed_of_light = 20.0
permittivity = inv(speed_of_light^2)
equation = Trixi.GlmMultiFluid5MomentPlasmaEquationsEntropy3D((1.6, 1.6), (25.0, 1.0), (-25.0, 1.0), speed_of_light, permittivity)
coordinates_min = (0.0, 0.0, 0.0)
coordinates_max = (1.0, 1.0, 1.0)

trees_per_dimension = (128, 128, 1)
basis = LobattoLegendreBasis(2)
mesh = P4estMesh(trees_per_dimension, polydeg = 1,
                 coordinates_min = coordinates_min, coordinates_max = coordinates_max,
                 periodicity = true, initial_refinement_level = 0)
indicator_sc = IndicatorHennemannGassner(equation, basis,
                                         alpha_max = 0.2,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = Trixi.flux_lax_friedrichs)
solver = DGSEM(basis, surface_flux, volume_integral)
semi = SemidiscretizationHyperbolic(mesh, equation,
                                    initial_condition_orszag_tang, solver, source_terms = Trixi.source_term_lorentz_corrected_2)

###############################################################################
# ODE solvers, callbacks etc.
amr_indicator = IndicatorHennemannGassner(semi,
                                          alpha_max = 0.5,
                                          alpha_min = 0.001,
                                          alpha_smooth = false,
                                          variable = density_pressure)
amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level = 0,
                                      max_level = 2, max_threshold = 0.01)
amr_callback = AMRCallback(semi, amr_controller,
                           interval = 6,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true)
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     save_analysis = true,
                                     output_directory = "out4"
                                     )
save_solution = SaveSolutionCallback(interval = 50,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim,
                                     output_directory = "out4"
                                    )
cfl = 1.0
tspan = (0.0, 0.5)

ode = semidiscretize(semi, tspan)
summary_callback = SummaryCallback()
stepsize_callback = StepsizeCallback(cfl = cfl)
callbacks = CallbackSet(summary_callback, analysis_callback, save_solution)

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
            abstol = 1.0e-6, reltol = 1.0e-6,
            ode_default_options()..., callback = callbacks);
