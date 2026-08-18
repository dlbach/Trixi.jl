using OrdinaryDiffEqLowStorageRK
using Trixi
using Random

###############################################################################
# semidiscretization of the Maxwell equations

function initial_condition_convergence(x, t, equations::Trixi.GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D)
    u1 = 2 + cos(2*pi*x[1])
    u2 = cos(2*pi*x[1])*sin(2*pi*t)
    u3 = cos(2*pi*x[2])
    u4 = cos(2*pi*x[1])
    u5 = 2 + sin(2*pi*x[2])

    u6 = sin(2*pi*x[1])*cos(2*pi*x[2])*sin(2*pi*t)
    u7 = -cos(2*pi*x[1])*sin(2*pi*x[2])*sin(2*pi*t)
    u8 = cos(2*pi*x[1])*sin(2*pi*t)
    u9 = sin(2*pi*x[1])*cos(2*pi*t)
    u10 = sin(2*pi*x[1])*cos(2*pi*x[2])*sin(2*pi*t)
    u11 = -cos(2*pi*x[1])*sin(2*pi*x[2])*sin(2*pi*t)
    u12 = cos(2*pi*x[2])*sin(2*pi*t)
    u13 = sin(2*pi*x[2])*cos(2*pi*t)
    return SVector(u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13)
end

function source_terms_convergence(u, x, t, equations::Trixi.GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D)
    gamma = equations.gammas[1]
    gm1 = gamma - 1
    c_sqr = equations.c_sqr
    c_e = equations.c_e
    c_b = equations.c_b

    rho = 2 + cos(2*pi*x[1])
    rho_v1 = cos(2*pi*x[1])*sin(2*pi*t)
    rho_v2 = cos(2*pi*x[2])
    rho_v3 = cos(2*pi*x[1])
    rho_s = 2 + sin(2*pi*x[2])

    rho_x = -2*pi*sin(2*pi*x[1])
    rho_v1_t = 2*pi*cos(2*pi*x[1])*cos(2*pi*t)
    rho_v1_x = -2*pi*sin(2*pi*x[1])*sin(2*pi*t)
    rho_v2_y = -2*pi*sin(2*pi*x[2])
    rho_v3_x = -2*pi*sin(2*pi*x[1])
    rho_s_y = 2*pi*cos(2*pi*x[2])

    v_1 = rho_v1 / rho
    v_2 = rho_v2 / rho
    v_3 = rho_v3 / rho 
    s = rho_s / rho
    p = rho^gamma * exp(s)


    s_x = - s * rho_x / rho
    s_y = rho_s_y / rho
    v_1_x = (rho_v1_x - v_1 * rho_x) / rho
    v_2_x = -v_2 * rho_x / rho
    v_2_y = rho_v2_y / rho
    v_3_x = (rho_v3_x - v_3 * rho_x) / rho
    p_x = exp(s) * (gamma * rho^gm1 * rho_x + rho^gamma * s_x)
    p_y = exp(s) * rho^gamma * s_y

    s1 = rho_v1_x + rho_v2_y
    s2 = v_1_x * rho_v1 + v_1 * rho_v1_x + v_1 * rho_v2_y + p_x + rho_v1_t
    s3 = v_2_x * rho_v1 + v_2 * rho_v1_x + v_2 * rho_v2_y + v_2_y * rho_v2 + p_y
    s4 = v_3_x * rho_v1 + v_3 * rho_v1_x + v_3 * rho_v2_y
    s5 = rho_s * (v_1_x + v_2_y) + rho_s_y * v_2

    E1 = sin(2*pi*x[1])*cos(2*pi*x[2])*sin(2*pi*t)
    E2 = -cos(2*pi*x[1])*sin(2*pi*x[2])*sin(2*pi*t)
    E3 = cos(2*pi*x[1])*sin(2*pi*t)
    B1 = sin(2*pi*x[1])*cos(2*pi*t)
    B2 = sin(2*pi*x[1])*cos(2*pi*x[2])*sin(2*pi*t)
    B3 = -cos(2*pi*x[1])*sin(2*pi*x[2])*sin(2*pi*t)
    psi_E = cos(2*pi*x[2])*sin(2*pi*t)
    psi_B = sin(2*pi*x[2])*cos(2*pi*t)

    E1_t = 2*pi*sin(2*pi*x[1])*cos(2*pi*x[2])*cos(2*pi*t)
    E1_x = 2*pi*cos(2*pi*x[1])*cos(2*pi*x[2])*sin(2*pi*t)
    E1_y = -2*pi*sin(2*pi*x[1])*sin(2*pi*x[2])*sin(2*pi*t)
    E2_t = -2*pi*cos(2*pi*x[1])*sin(2*pi*x[2])*cos(2*pi*t)
    E2_x = 2*pi*sin(2*pi*x[1])*sin(2*pi*x[2])*sin(2*pi*t)
    E2_y = -2*pi*cos(2*pi*x[1])*cos(2*pi*x[2])*sin(2*pi*t)
    E3_t = 2*pi*cos(2*pi*x[1])*cos(2*pi*t)
    E3_x = -2*pi*sin(2*pi*x[1])*sin(2*pi*t)

    B1_t = -2*pi*sin(2*pi*x[1])*sin(2*pi*t)
    B1_x = 2*pi*cos(2*pi*x[1])*cos(2*pi*t)
    B2_t = 2*pi*sin(2*pi*x[1])*cos(2*pi*x[2])*cos(2*pi*t)
    B2_x = 2*pi*cos(2*pi*x[1])*cos(2*pi*x[2])*sin(2*pi*t)
    B2_y = -2*pi*sin(2*pi*x[1])*sin(2*pi*x[2])*sin(2*pi*t)
    B3_t = -2*pi*cos(2*pi*x[1])*sin(2*pi*x[2])*cos(2*pi*t)
    B3_x = 2*pi*sin(2*pi*x[1])*sin(2*pi*x[2])*sin(2*pi*t)
    B3_y = -2*pi*cos(2*pi*x[1])*cos(2*pi*x[2])*sin(2*pi*t)

    psi_E_t = 2*pi*cos(2*pi*x[2])*cos(2*pi*t)
    psi_E_y = -2*pi*sin(2*pi*x[2])*sin(2*pi*t)
    psi_B_t = -2*pi*sin(2*pi*x[2])*sin(2*pi*t)
    psi_B_y = 2*pi*cos(2*pi*x[2])*cos(2*pi*t)

    s6 = E1_t - c_sqr * B3_y
    s7 = E2_t + c_sqr * (B3_x + c_e * psi_E_y)
    s8 = E3_t + c_sqr * (-B2_x)
    s9 = B1_t
    s10 = B2_t - E3_x + c_b * psi_B_y
    s11 = B3_t + E2_x - E1_y
    s12 = psi_E_t + c_e * (E1_x + E2_y)
    s13 = psi_B_t + c_sqr * c_b * (B1_x + B2_y)

    return SVector(s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13)
end


volume_flux = (Trixi.flux_energy_central, Trixi.flux_noncon_empty)
surface_flux = (Trixi.flux_energy_upwind, FluxPlusDissipation(Trixi.flux_noncon_empty, DissipationMatrixWintersEtal()))
equation = Trixi.GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D(1.4, 1.0, 1.0, 10.0, 0.01)
mesh = TreeMesh((-1.0, -1.0), (1.0, 1.0), periodicity = true, initial_refinement_level = 2, n_cells_max = 10^7)
solver = DGSEM(polydeg = 3, surface_flux = surface_flux, volume_integral = VolumeIntegralFluxDifferencing(volume_flux))
semi = SemidiscretizationHyperbolic(mesh, equation,
                                    initial_condition_convergence, solver, source_terms = source_terms_convergence)

###############################################################################
# ODE solvers, callbacks etc.

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     save_analysis = true)

cfl = 0.5
tspan = (0.0, 0.01)

ode = semidiscretize(semi, tspan)
summary_callback = SummaryCallback()
stepsize_callback = StepsizeCallback(cfl = cfl)
callbacks = CallbackSet(summary_callback, analysis_callback, stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
