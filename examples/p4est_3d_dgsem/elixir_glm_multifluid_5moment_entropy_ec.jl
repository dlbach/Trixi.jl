using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the Maxwell equations

@inline function initial_condition_ec(x, t, equations::Trixi.GlmMultiFluid5MomentPlasmaEquationsEntropy3D)
    u = SVector(5.0, 0.0, 0.0, 0.0, -10.0, 5.0, 0.0, 0.0, 0.0, -10.0, 
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    v = rand(18) .- 0.5
    #v[7] /= equations.speed_of_light
    #v[8] /= equations.speed_of_light
    v = SVector{18, Float64}(v)
    return u + v
end

volume_flux = Trixi.flux_energy_upwind
surface_flux = Trixi.flux_energy_central
equation = Trixi.GlmMultiFluid5MomentPlasmaEquationsEntropy3D((1.6, 1.6), (1.0, 1.0), (1.0, -1.0), 1e2, 1e-2, 1e0)
coordinates_min = (-1.0, -1.0, -1.0)
coordinates_max = (1.0, 1.0, 1.0)

trees_per_dimension = (2, 2, 2)

mesh = P4estMesh(trees_per_dimension, polydeg = 2,
                 coordinates_min = coordinates_min, coordinates_max = coordinates_max,
                 periodicity = true, initial_refinement_level = 2)
basis = LobattoLegendreBasis(2)

indicator_sc = IndicatorHennemannGassner(equation, basis,
                                         alpha_max = 0.5,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)

solver = DGSEM(basis, surface_flux, VolumeIntegralFluxDifferencing(volume_flux))
semi = SemidiscretizationHyperbolic(mesh, equation,
                                    initial_condition_ec, solver, source_terms = Trixi.source_term_lorentz_corrected_2)

###############################################################################
# ODE solvers, callbacks etc.

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     save_analysis = true, output_directory="out",
                                     analysis_filename="analysis.dat")

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
