using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the Maxwell equations

function initial_condition_orszag_tang(x, t, equations::Trixi.GlmMultiFluid5MomentPlasmaEquations3D)
    gamma_i = equations.gammas[2]
    gamma_e = equations.gammas[1]
    inv_gm1_i = equations.inv_gammas_minus_one[2]
    inv_gm1_e = equations.inv_gammas_minus_one[1]
    R_i = equations.gas_constants[2]
    R_e = equations.gas_constants[1]
    delta_B = 0.2
    delta_u = 0.2
    p_i = R_i * 0.05
    p_e = R_e * 0.05 * 0.04
    B_0 = 1.0 #sqrt(2 * p_i)

    rho_i = 1.0
    rho_e = 0.04 * (1 + 2 * pi * delta_u * B_0 * (cospi(2 * x[1]) + cospi(2 * x[2])) * equations.permittivity / equations.c_e)
    rho_v1_i = -delta_u * sinpi(2 * x[2]) * rho_i
    rho_v2_i = delta_u * sinpi(2 * x[1]) * rho_i
    rho_v3_i = 0.0
    rho_v1_e = -delta_u * sinpi(2 * x[2]) * rho_e
    rho_v2_e = delta_u * sinpi(2 * x[1]) * rho_e
    rho_v3_e = -2 * pi * delta_u * (2 * cospi(4 * x[1]) + cospi(2 * x[2])) * rho_e
    rho_e_total_i = 0.5 * (rho_v1_i^2 + rho_v2_i^2 + rho_v3_i^2) / rho_i + p_i * inv_gm1_i
    rho_e_total_e = 0.5 * (rho_v1_e^2 + rho_v2_e^2 + rho_v3_e^2) / rho_e + p_e * inv_gm1_e

    E1 = -delta_u * B_0 * sinpi(2 * x[1]) 
    E2 = -delta_u * B_0 * sinpi(2 * x[2])
    E3 = 0.0
    B1 = -delta_B * B_0 * sinpi(2 * x[2])
    B2 = delta_B * B_0 * sinpi(4 * x[1])
    B3 = B_0
    psi_E = 0.0
    psi_B = 0.0

    return SVector(rho_e, rho_v1_e, rho_v2_e, rho_v3_e, rho_e_total_e, rho_i, rho_v1_i, rho_v2_i, rho_v3_i, rho_e_total_i, E1, E2, E3, B1, B2, B3, psi_E, psi_B)
end

volume_flux = Trixi.flux_ranocha_central
surface_flux = Trixi.flux_ranocha_upwind
speed_of_light = 18.172
permittivity = inv(speed_of_light^2)
equation = Trixi.GlmMultiFluid5MomentPlasmaEquations3D((1.6, 1.6), (25.0, 1.0), (-25.0, 1.0), speed_of_light, permittivity)
coordinates_min = (0.0, 0.0, 0.0)
coordinates_max = (1.0, 1.0, 1.0)

trees_per_dimension = (64, 64, 1)
basis = LobattoLegendreBasis(3)
mesh = P4estMesh(trees_per_dimension, polydeg = 1,
                 coordinates_min = coordinates_min, coordinates_max = coordinates_max,
                 periodicity = true, initial_refinement_level = 0)
indicator_sc = IndicatorHennemannGassner(equation, basis,
                                         alpha_max = 0.5,
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

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     save_analysis = true, output_directory="out2",
                                     analysis_filename="analysis.dat",
                                    )   
save_solution = SaveSolutionCallback(interval = 50,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim,
                                     output_directory = "out2"
                                    )
cfl = 1.0
tspan = (0.0, 1.0)

ode = semidiscretize(semi, tspan)
summary_callback = SummaryCallback()
stepsize_callback = StepsizeCallback(cfl = cfl)
callbacks = CallbackSet(summary_callback, analysis_callback, stepsize_callback, save_solution)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
