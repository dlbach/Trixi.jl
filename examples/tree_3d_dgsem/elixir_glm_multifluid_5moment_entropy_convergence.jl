using OrdinaryDiffEqLowStorageRK
using Trixi
using Random

###############################################################################
# semidiscretization of the Maxwell equations

function initial_condition_convergence(x, t, equations::Trixi.GlmMultiFluid5MomentPlasmaEquationsEntropy2D)
    u1 = 10 + cos(2*pi*x[1])
    u2 = cos(2*pi*x[1])*sin(2*pi*t)
    u3 = cos(2*pi*x[2])
    u4 = -1 + sin(2*pi*x[2])
    u5 = sin(2*pi*x[1])*cos(2*pi*x[2])*sin(2*pi*t)
    u6 = -cos(2*pi*x[1])*sin(2*pi*x[2])*sin(2*pi*t)
    u7 = cos(2*pi*x[1])*sin(2*pi*t)#/equations.speed_of_light
    u8 = sin(2*pi*x[1])*cos(2*pi*t)#/equations.speed_of_light
    return SVector(u1, u2, u3, u4, u5, u6, u7, u8)
end

function source_terms_convergence(u, x, t, equations::Trixi.GlmMultiFluid5MomentPlasmaEquationsEntropy2D)
    gamma = equations.gammas[1]
    gm1 = gamma - 1
    rho_s = -1 + sin(2*pi*x[2])
    rho = 10 + cos(2*pi*x[1])
    rho_x = -2*pi*sin(2*pi*x[1])
    rho_v1 = cos(2*pi*x[1])*sin(2*pi*t)
    rho_v2 = cos(2*pi*x[2])
    rho_v1_t = 2*pi*cos(2*pi*x[1])*cos(2*pi*t)
    rho_v1_x = -2*pi*sin(2*pi*x[1])*sin(2*pi*t)
    rho_v2_y = -2*pi*sin(2*pi*x[2])
    rho_s_y = 2*pi*cos(2*pi*x[2])
    s = rho_s / rho
    v_1 = rho_v1 / rho
    v_2 = rho_v2 / rho
    s_x = -s * rho_x / rho
    s_y = rho_s_y / rho
    v_1_x = (rho_v1_x - v_1 * rho_x) / rho #2*pi*sin(2*pi*t)*sin(2*pi*x[1]) / rho^2
    v_2_x = -v_2 * rho_x / rho #2*pi*sin(2*pi*x[1])*cos(2*pi*x[2]) / rho^2
    v_2_y = rho_v2_y / rho #-2*pi*sin(2*pi*x[2]) / rho
    p = rho^gamma * exp(s)
    p_x = exp(s) * (gamma * rho^gm1 * rho_x + rho^gamma * s_x)
    p_y = exp(s) * rho^gamma * s_y

    s1 = rho_v1_x + rho_v2_y
    s2 = p_x + 2 * v_1 * rho_v1_x - v_1^2 * rho_x + v_1 * rho_v2_y + rho_v1_t
    s3 = v_2 * rho_v1_x - v_1*v_2*rho_x + 2*v_2*rho_v2_y + p_y
    s4 = rho_s * (v_1_x + v_2_y) + v_2 * rho_s_y
    s5 = 2*pi*(sin(2*pi*x[1])*cos(2*pi*x[2])*cos(2*pi*t) + cos(2*pi*x[1])*cos(2*pi*t)*equations.speed_of_light^2)
    s6 = -2*pi*(cos(2*pi*x[1])*sin(2*pi*x[2])*cos(2*pi*t) + sin(2*pi*x[1])*sin(2*pi*t)*equations.speed_of_light^2)
    s7 = 2*pi*cos(2*pi*x[1])*cos(2*pi*t) + 4*pi*sin(2*pi*x[1])*sin(2*pi*x[2])*sin(2*pi*t)
    #s7 = 2*pi*cos(2*pi*x[1])*cos(2*pi*t)/equations.speed_of_light + 4*pi*sin(2*pi*x[1])*sin(2*pi*x[2])*sin(2*pi*t)
    s8 = -2*pi*sin(2*pi*x[1])*sin(2*pi*t)#/equations.speed_of_light
    return SVector(s1, s2, s3, s4, s5, s6, s7, s8)
end


volume_flux = Trixi.flux_energy_central
equation = Trixi.GlmMultiFluid5MomentPlasmaEquationsEntropy2D(1.4, 1.0, 1.0, 1e6, 1e-22)
mesh = TreeMesh((-1.0, -1.0), (1.0, 1.0), periodicity = true, initial_refinement_level = 2, n_cells_max = 10^7)
solver = DGSEM(polydeg = 3, surface_flux = Trixi.flux_energy_upwind, volume_integral = VolumeIntegralFluxDifferencing(volume_flux))
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
