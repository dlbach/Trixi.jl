using OrdinaryDiffEqLowStorageRK
using Trixi
using Random

###############################################################################
# semidiscretization of the Maxwell equations

function initial_condition_orszag_tang(x, t, equations::Trixi.GlmMultiFluid5MomentPlasmaEquationsEntropy3D)
    
    return SVector(u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16, u17, u18)
end

volume_flux = Trixi.flux_energy_central
equation = Trixi.GlmMultiFluid5MomentPlasmaEquationsEntropy3D((1.6, 1.6), (1.0, 1.0), (-1.0, 1.0), 1.0, 1e6, 1e-22)
mesh = TreeMesh((0.0, 0.0, -1.0), (4*pi, 4*pi, 1.0), initial_refinement_level = 2, periodicity = true, n_cells_max = 10^7)
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
