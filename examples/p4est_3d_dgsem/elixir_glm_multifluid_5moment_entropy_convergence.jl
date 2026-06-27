using OrdinaryDiffEqLowStorageRK
using Trixi
using Random

###############################################################################
# semidiscretization of the Maxwell equations

function initial_condition_convergence(x, t, equations::Trixi.GlmMultiFluid5MomentPlasmaEquationsEntropy3D)
    u1 = 10 + cos(2*pi*x[1])*cos(2*pi*x[3])
    u2 = cos(2*pi*x[1])*sin(2*pi*t)
    u3 = cos(2*pi*x[2])*cos(2*pi*x[2])
    u4 = cos(2*pi*x[1])*cos(2*pi*x[3])
    u5 = -1 + sin(2*pi*x[2])*sin(2*pi*x[3])

    u6 = sin(2*pi*x[1])*cos(2*pi*x[2])*sin(2*pi*t)
    u7 = -cos(2*pi*x[1])*sin(2*pi*x[2])*sin(2*pi*t)
    u8 = cos(2*pi*x[1])*sin(2*pi*t)*cos(2*pi*x[3])
    u9 = sin(2*pi*x[1])*cos(2*pi*t)*cos(2*pi*x[3])
    u10 = sin(2*pi*x[1])*cos(2*pi*x[2])*sin(2*pi*t)
    u11 = -cos(2*pi*x[1])*sin(2*pi*x[2])*sin(2*pi*t)
    u12 = cos(2*pi*x[2])*sin(2*pi*t)*cos(2*pi*x[3])
    u13 = sin(2*pi*x[2])*cos(2*pi*t)
    return SVector(u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13)
end

function source_terms_convergence(u, x, t, equations::Trixi.GlmMultiFluid5MomentPlasmaEquationsEntropy3D)
    gamma = equations.gammas[1]
    gm1 = gamma - 1
    c_sqr = equations.c_sqr
    c_e = equations.c_e
    c_b = equations.c_b

    rho = 10 + cos(2*pi*x[1])*cos(2*pi*x[3])
    rho_v1 = cos(2*pi*x[1])*sin(2*pi*t)
    rho_v2 = cos(2*pi*x[2])*cos(2*pi*x[3])
    rho_v3 = cos(2*pi*x[1])*cos(2*pi*x[3])
    rho_s = -1 + sin(2*pi*x[2])*sin(2*pi*x[3])

    rho_x = -2*pi*sin(2*pi*x[1])*cos(2*pi*x[3])
    rho_z = -2*pi*cos(2*pi*x[1])*sin(2*pi*x[3])
    rho_v1_t = 2*pi*cos(2*pi*x[1])*cos(2*pi*t)
    rho_v1_x = -2*pi*sin(2*pi*x[1])*sin(2*pi*t)
    rho_v2_y = -2*pi*sin(2*pi*x[2])*cos(2*pi*x[3])
    rho_v2_z = -2*pi*cos(2*pi*x[2])*sin(2*pi*x[3])
    rho_v3_x = -2*pi*sin(2*pi*x[1])*cos(2*pi*x[3])
    rho_v3_z = -2*pi*cos(2*pi*x[1])*sin(2*pi*x[3])
    rho_s_y = 2*pi*cos(2*pi*x[2])*sin(2*pi*x[3])
    rho_s_z = 2*pi*sin(2*pi*x[2])*cos(2*pi*x[3])

    v_1 = rho_v1 / rho
    v_2 = rho_v2 / rho
    v_3 = rho_v2 / rho 
    s = rho_s / rho
    p = rho^gamma * exp(s)


    s_x = - s * rho_x / rho
    s_y = rho_s_y / rho
    s_z = (rho_s_z - s * rho_z) / rho
    v_1_t = rho_v1_t / rho
    v_1_x = (rho_v1_x - v_1 * rho_x) / rho
    v_1_z = -v_1 * rho_z / rho
    v_2_x = -v_2 * rho_x / rho
    v_2_y = rho_v2_y / rho
    v_2_z = (rho_v2_z - v_2 * rho_z) / rho
    v_3_x = (rho_v3_x - v_3 * rho_x) / rho
    v_3_z = (rho_v3_z - v_3 * rho_z) / rho
    p_x = exp(s) * (gamma * rho^gm1 * rho_x + rho^gamma * s_x)
    p_y = exp(s) * rho^gamma * s_y
    p_z = exp(s) * (gamma * rho^gm1 * rho_z + rho^gamma * s_z)

    s1 = rho_v1_x + rho_v2_y + rho_v3_z
    s2 = v_1_x * rho_v1 + v_1 * rho_v1_x + v_1 * rho_v2_y + v_1_z * rho_v3 + v_1 * rho_v3_z + p_x + rho_v1_t
    s3 = v_2_x * rho_v1 + v_2 * rho_v1_x + v_2 * rho_v2_y + v_2_y * rho_v2 + v_2_z * rho_v3 + v_2 * rho_v3_z + p_y
    s4 = v_3_x * rho_v1 + v_3 * rho_v1_x + v_3 * rho_v2_y + v_3_z * rho_v3 + v_3 * rho_v3_z + p_z
    s5 = rho_s * (v_1_x + v_2_y + v_3_z) + rho_s_y * v_2 + rho_s_z * v_3

    E1 = sin(2*pi*x[1])*cos(2*pi*x[2])*sin(2*pi*t)
    E2 = -cos(2*pi*x[1])*sin(2*pi*x[2])*sin(2*pi*t)
    E3 = cos(2*pi*x[1])*sin(2*pi*t)*cos(2*pi*x[3])
    B1 = sin(2*pi*x[1])*cos(2*pi*t)*cos(2*pi*x[3])
    B2 = sin(2*pi*x[1])*cos(2*pi*x[2])*sin(2*pi*t)
    B3 = -cos(2*pi*x[1])*sin(2*pi*x[2])*sin(2*pi*t)
    psi_E = cos(2*pi*x[2])*sin(2*pi*t)*cos(2*pi*x[3])
    psi_B = sin(2*pi*x[2])*cos(2*pi*t)

    E1_t = 2*pi*sin(2*pi*x[1])*cos(2*pi*x[2])*cos(2*pi*t)
    E1_x = 2*pi*cos(2*pi*x[1])*cos(2*pi*x[2])*sin(2*pi*t)
    E1_y = -2*pi*sin(2*pi*x[1])*sin(2*pi*x[2])*sin(2*pi*t)
    E2_t = -2*pi*cos(2*pi*x[1])*sin(2*pi*x[2])*cos(2*pi*t)
    E2_x = 2*pi*sin(2*pi*x[1])*sin(2*pi*x[2])*sin(2*pi*t)
    E2_y = -2*pi*cos(2*pi*x[1])*cos(2*pi*x[2])*sin(2*pi*t)
    E3_t = 2*pi*cos(2*pi*x[1])*cos(2*pi*t)*cos(2*pi*x[3])
    E3_x = -2*pi*sin(2*pi*x[1])*sin(2*pi*t)*cos(2*pi*x[3])
    E3_z = -2*pi*cos(2*pi*x[1])*sin(2*pi*t)*sin(2*pi*x[3])

    B1_t = -2*pi*sin(2*pi*x[1])*sin(2*pi*t)*cos(2*pi*x[3])
    B1_x = 2*pi*cos(2*pi*x[1])*cos(2*pi*t)*cos(2*pi*x[3])
    B1_z = -2*pi*sin(2*pi*x[1])*cos(2*pi*t)*sin(2*pi*x[3])
    B2_t = 2*pi*sin(2*pi*x[1])*cos(2*pi*x[2])*cos(2*pi*t)
    B2_x = 2*pi*cos(2*pi*x[1])*cos(2*pi*x[2])*sin(2*pi*t)
    B2_y = -2*pi*sin(2*pi*x[1])*sin(2*pi*x[2])*sin(2*pi*t)
    B3_t = -2*pi*cos(2*pi*x[1])*sin(2*pi*x[2])*cos(2*pi*t)
    B3_x = 2*pi*sin(2*pi*x[1])*sin(2*pi*x[2])*sin(2*pi*t)
    B3_y = -2*pi*cos(2*pi*x[1])*cos(2*pi*x[2])*sin(2*pi*t)

    psi_E_t = 2*pi*cos(2*pi*x[2])*cos(2*pi*t)*cos(2*pi*x[3])
    psi_E_y = -2*pi*sin(2*pi*x[2])*sin(2*pi*t)*cos(2*pi*x[3])
    psi_E_z = -2*pi*cos(2*pi*x[2])*sin(2*pi*t)*sin(2*pi*x[3])
    psi_B_t = -2*pi*sin(2*pi*x[2])*sin(2*pi*t)
    psi_B_y = 2*pi*cos(2*pi*x[2])*cos(2*pi*t)

    s6 = E1_t - c_sqr * B3_y
    s7 = E2_t + c_sqr * (B3_x + c_e * psi_E_y - B1_z)
    s8 = E3_t + c_sqr * (-B2_x + c_e * psi_E_z)
    s9 = B1_t - E2_y
    s10 = B2_t - E3_x + c_b * psi_B_y
    s11 = B3_t + E2_x - E1_y
    s12 = psi_E_t + c_e * (E1_x + E2_y + E3_z)
    s13 = psi_B_t + c_sqr * c_b * (B1_x + B2_y)

    return SVector(s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13)
end


volume_flux = Trixi.flux_energy_central
equation = Trixi.GlmMultiFluid5MomentPlasmaEquationsEntropy3D(1.4, 1.0, 1.0, 1e6, 1e-22)
coordinates_min = (-1.0, -1.0, -1.0)
coordinates_max = (1.0, 1.0, 1.0)

trees_per_dimension = (1, 1, 1)

mesh = P4estMesh(trees_per_dimension, polydeg = 2,
                 coordinates_min = coordinates_min, coordinates_max = coordinates_max,
                 periodicity = true, initial_refinement_level = 1)
solver = DGSEM(polydeg = 3, surface_flux = Trixi.flux_energy_diss_upwind, volume_integral = VolumeIntegralFluxDifferencing(volume_flux))
semi = SemidiscretizationHyperbolic(mesh, equation,
                                    initial_condition_convergence, solver, source_terms = source_terms_convergence)

###############################################################################
# ODE solvers, callbacks etc.

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     save_analysis = true)

cfl = 0.5
tspan = (0.0, 1e-7)

ode = semidiscretize(semi, tspan)
summary_callback = SummaryCallback()
stepsize_callback = StepsizeCallback(cfl = cfl)
callbacks = CallbackSet(summary_callback, analysis_callback, stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
