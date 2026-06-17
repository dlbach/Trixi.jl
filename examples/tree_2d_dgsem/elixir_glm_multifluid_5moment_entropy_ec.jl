using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the Maxwell equations

@inline function initial_condition_ec(x, t, equations::Trixi.GlmMultiFluid5MomentPlasmaEquationsEntropy2D)
    u = SVector(5.0, 0.0, 0.0, -10.0, 0.0, 0.0, 0.0, 0.0)
    v = 2.0 .* rand(8) .- 2.0
    #v[7] /= equations.speed_of_light
    #v[8] /= equations.speed_of_light
    v = SVector{8, Float64}(v)
    return u + v
end

volume_flux = Trixi.flux_energy_central
surface_flux = Trixi.flux_energy_central
equation = Trixi.GlmMultiFluid5MomentPlasmaEquationsEntropy2D(1.4, 1.0, 1.0, 1e2, 1e-2, 1e0)
mesh = TreeMesh((-1.0, -1.0), (1.0, 1.0), initial_refinement_level = 2, n_cells_max = 10^4)
basis = LobattoLegendreBasis(2)
volume_integral = VolumeIntegralFluxDifferencing(volume_flux)
solver = DGSEM(basis, surface_flux, volume_integral)
semi = SemidiscretizationHyperbolic(mesh, equation,
                                    initial_condition_ec, solver, source_terms = Trixi.source_term_lorentz_corrected)

###############################################################################
# ODE solvers, callbacks etc.

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     save_analysis = true)

cfl = 0.0001
tspan = (0.0, 1.0)

ode = semidiscretize(semi, tspan)
summary_callback = SummaryCallback()
stepsize_callback = StepsizeCallback(cfl = cfl)
callbacks = CallbackSet(summary_callback, analysis_callback, stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
