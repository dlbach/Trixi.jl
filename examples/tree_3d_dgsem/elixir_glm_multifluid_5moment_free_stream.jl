using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the Maxwell equations

function initial_condition_constant(x, t, equations::Trixi.GlmMultiFluid5MomentPlasmaEquations3D)
    return SVector(1.0, 0.2, -0.5, 0.7, 10.0, 1.0, 0.2, -0.5, 0.6, 10.0, 
                   2.0, 3.0, -0.5, 4.0/equations.c_sqr, 5.0/equations.c_sqr,
                   -3.0/equations.c_sqr, 4.0/equations.c_sqr, 0.3)
end

equation = Trixi.GlmMultiFluid5MomentPlasmaEquations3D((1.6, 1.6), (1.0, 1.0), (1.0, -1.0), 20.0, 10.0)
mesh = TreeMesh((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), periodicity = true, initial_refinement_level = 2, 
                n_cells_max = 10^4, refinement_patches = refinement_patches)
volume_flux = Trixi.flux_ranocha_central
surface_flux = Trixi.flux_ranocha_upwind
basis = LobattoLegendreBasis(4)
indicator_sc = IndicatorHennemannGassner(equation, basis,
                                         alpha_max = 0.5,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)

solver = DGSEM(basis, surface_flux, volume_integral)
semi = SemidiscretizationHyperbolic(mesh, equation,
                                    initial_condition_constant, solver)

###############################################################################
# ODE solvers, callbacks etc.

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     save_analysis = true)

cfl = 0.5
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
