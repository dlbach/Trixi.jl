# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.

@muladd begin
#! format: noindent

struct GlmMultiFluid5MomentPlasmaEquationsEntropy2D{NVARS, NCOMP, RealT <: Real} <: AbstractGlmMultiFluid5MomentPlasmaEquationsEntropy{2, NVARS, NCOMP}
    gammas::SVector{NCOMP, RealT}               # specific heat ratio for each species
    inv_gammas_minus_one::SVector{NCOMP, RealT}  # = inv(gamma - 1); can be used to write slow divisions as fast multiplications
    gas_constants::SVector{NCOMP, RealT}        # specific gas constant for each species
    charge_mass_ratios::SVector{NCOMP, RealT}      # charge of one particle of each species divided by its mass
    speed_of_light::RealT                       # c
    c_sqr::RealT                                # squared speed of light
    permittivity::RealT                         # absolute dielectric permittivity
    permeability::RealT                         # magnetic permeability
    c_e::RealT                                  # GLM cleaning speed for the electric field
    function GlmMultiFluid5MomentPlasmaEquationsEntropy2D{NVARS, NCOMP, RealT}(gammas::SVector{NCOMP, RealT},
                                                                    gas_constants::SVector{NCOMP, RealT},
                                                                    charge_mass_ratios::SVector{NCOMP, RealT},
                                                                    speed_of_light::RealT, permittivity::RealT,
                                                                    c_e::RealT) where {
                                                                                                                    NVARS,
                                                                                                                    NCOMP,
                                                                                                                    RealT <:
                                                                                                                    Real
                                                                                                                    }
        NCOMP >= 1 ||
            throw(DimensionMismatch("`gammas` and `charge_per_mass` have to be filled with at least one value"))

        inv_gammas_minus_one = inv.(gammas .- 1)
        permeability = inv(speed_of_light^2 * permittivity)
        new(gammas, inv_gammas_minus_one, gas_constants, charge_mass_ratios, speed_of_light, speed_of_light^2, permittivity, permeability, c_e)
    end
end

function GlmMultiFluid5MomentPlasmaEquationsEntropy2D(gammas, gas_constants, charge_mass_ratios, speed_of_light = 299_792_458.0, 
                                            permittivity = 8.8541878188e-12, c_e = 1.0)
    _gammas = promote(gammas...)
    _gas_constants = promote(gas_constants...)
    _charge_mass_ratios = promote(charge_mass_ratios...)

    RealT = promote_type(eltype(_gammas), eltype(_gas_constants), eltype(_charge_mass_ratios), typeof(speed_of_light), typeof(permittivity), typeof(c_e))
    
    _gammas = SVector(map(RealT, _gammas))
    _gas_constants = SVector(map(RealT, _gas_constants))
    _charge_mass_ratios = SVector(map(RealT, _charge_mass_ratios))

    speed_of_light = convert(RealT, speed_of_light)
    permittivity = convert(RealT, permittivity)
    c_e = convert(RealT, c_e)

    NVARS = 4*length(_gammas) + 4
    NCOMP = length(_gammas)
    return GlmMultiFluid5MomentPlasmaEquationsEntropy2D{NVARS, NCOMP, RealT}(_gammas, _gas_constants, _charge_mass_ratios, speed_of_light, permittivity, c_e)
end
    
function varnames(::typeof(cons2cons), equations::GlmMultiFluid5MomentPlasmaEquationsEntropy2D)
    fluids = ntuple(n -> SVector("rho_" * string(n), "rho_v1_" * string(n), "rho_v2_" * string(n), "rho_s_" * string(n)), Val(ncomponents(equations)))
    glm = ("E1", "E2", "B", "psi_E")
    return (reduce(vcat, fluids)..., glm...)
end

@inline function Base.real(::GlmMultiFluid5MomentPlasmaEquationsEntropy2D{NVARS, NCOMP, RealT}) where {
                                                                                               NVARS,
                                                                                               NCOMP,
                                                                                               RealT
                                                                                               }
    return RealT
end

# Convert conservative vaiables to primitive
@inline function cons2prim(u, equations::GlmMultiFluid5MomentPlasmaEquationsEntropy2D)
    prims_euler = SVector(ntuple(i -> cons2prim_euler(u, i, equations), ncomponents(equations)))
    prims_glm = SVector(u[end-3], u[end-2], u[end-1], u[end])
    return vcat(reduce(vcat, prims_euler), prims_glm)
end

@inline function cons2prim_euler(u, i, equations::GlmMultiFluid5MomentPlasmaEquationsEntropy2D)
    rho, rho_v1, rho_v2, rho_s = view(u, (4*i-3):(4*i))
    gamma = equations.gammas[i]

    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    s = rho_s / rho
    p = rho^gamma * exp(s)

    return SVector(rho, v1, v2, p)
end

@inline function prim2cons(prim, equations::GlmMultiFluid5MomentPlasmaEquationsEntropy2D)
    prims_euler = SVector(ntuple(i -> prim2cons_euler(prim, i, equations), ncomponents(equations)))
    prims_glm = SVector(prim[end-3], prim[end-2], prim[end-1], prim[end])
    return vcat(reduce(vcat, prims_euler), prims_glm)
end

# Convert primitive to conservative variables
@inline function prim2cons_euler(prim, i, equations::GlmMultiFluid5MomentPlasmaEquationsEntropy2D)
    rho, v1, v2, p = view(prim, (4*i-3):(4*i))
    gamma = equations.gammas[i]

    rho_v1 = rho * v1
    rho_v2 = rho * v2
    rho_s = rho * log(p / rho^gamma)

    return SVector(rho, rho_v1, rho_v2, rho_e)
end

# Convert conservative variables to entropy variables
@inline function cons2entropy(u, equations::GlmMultiFluid5MomentPlasmaEquationsEntropy2D)
    entropy_euler = SVector(ntuple(i -> cons2entropy_euler(u, i, equations), ncomponents(equations)))
    entropy_glm = SVector(u[end-3]*equations.permittivity, u[end-2]*equations.permittivity, 
                        u[end-1]/equations.permeability, u[end]/equations.permeability)
    return vcat(reduce(vcat, entropy_euler), entropy_glm)
end

@inline function cons2entropy_euler(u, i, equations::GlmMultiFluid5MomentPlasmaEquationsEntropy2D)
    rho, rho_v1, rho_v2, rho_s = view(u, (4*i-3):(4*i))
    gamma = equations.gammas[i]
    inv_gamma_minus_one = equations.inv_gammas_minus_one[i]

    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    s = rho_s / rho
    p = rho^(gamma) * exp(s)
    v_square = v1^2 + v2^2
    p_div_rho = p / rho

    w = -0.5f0 * v_square + inv_gamma_minus_one * (gamma * p_div_rho - p_div_rho * s)

    return SVector(w, v1, v2, inv_gamma_minus_one * p_div_rho)
end

# Convert entropy variables to conservative variables
@inline function entropy2cons(w, equations::GlmMultiFluid5MomentPlasmaEquationsEntropy2D)
    cons_euler = SVector(ntuple(i -> entropy2cons_euler(w, i, equations), ncomponents(equations)))
    cons_glm = equations.T_min * SVector(w[end-3]/equations.permittivity, w[end-2]/equations.permittivity, 
                                        w[end-1]*equations.permeability, w[end]*equations.permeability)
    return vcat(reduce(vcat, cons_euler), cons_glm)
end

@inline function entropy2cons_euler(w, i, equations::GlmMultiFluid5MomentPlasmaEquationsEntropy2D)
    gamma = equations.gammas[i]
    inv_gamma_minus_one = equations.inv_gammas_minus_one[i]
    V1, V2, V3, V5 = view(w, (4*i-3):(4*i))
    
    v_square = V2^2 + V3^2
    p_div_rho =  (gamma - 1) * V5
    s = (V1 + 0.5f0 * v_square - gamma * rho_p / inv_gamma_minus_one) * (gamma - 1) / p_div_rho
    rho_gamma_minus_one = inv(exp(s) / p_div_rho)

    rho = exp(log(rho_gamma_minus_one) * inv_gamma_minus_one)
    rho_v1 = rho * V2
    rho_v2 = rho * V3
    rho_s = rho * s
    return SVector(rho, rho_v1, rho_v2, rho_s)
end

function default_analysis_integrals(::GlmMultiFluid5MomentPlasmaEquationsEntropy2D)
    (Val(:l2_dive), Val(:l2_e_normal_jump), entropy_timederivative)
end

@inline electric_field(u, equations::GlmMultiFluid5MomentPlasmaEquationsEntropy2D) = SVector(u[end-3], u[end-2])

@inline function charge_density(u, equations::GlmMultiFluid5MomentPlasmaEquationsEntropy2D) 
    return sum(densities(u, equations)[i] * equations.charge_mass_ratios[i] for i in 1:ncomponents(equations))
end

@inline function current_density(u, equations::GlmMultiFluid5MomentPlasmaEquationsEntropy2D) 
    return sum(momenta(u, equations)[i] * equations.charge_mass_ratios[i] for i in 1:ncomponents(equations))
end

@inline function scaled_charge_density(u, x, t, source_terms, equations::GlmMultiFluid5MomentPlasmaEquationsEntropy2D) 
    return charge_density(u, equations) / equations.permittivity
end

@inline function densities(u, equations::GlmMultiFluid5MomentPlasmaEquationsEntropy2D)
    return SVector(ntuple(i -> u[4*i-3], ncomponents(equations)))
end

@inline function momenta(u, equations::GlmMultiFluid5MomentPlasmaEquationsEntropy2D)
    return SVector(ntuple(i -> SVector(u[4*i-2], u[4*i-1]), ncomponents(equations)))
end

@inline function entropies(u, equations::GlmMultiFluid5MomentPlasmaEquationsEntropy2D)
    return SVector(ntuple(i -> u[4*i], ncomponents(equations)))
end

@inline function flux(u, orientation::Integer, equations::GlmMultiFluid5MomentPlasmaEquationsEntropy2D)
    fluxes_euler = ntuple(i -> flux_euler(u, orientation, i, equations), ncomponents(equations))
    flux_glm = flux_glm_maxwell(u, orientation, equations)
    return vcat(fluxes_euler..., flux_glm)
end

# Calculates the Euler flux for a single species at a single point
@inline function flux_euler(u, orientation::Integer, i::Integer, equations::GlmMultiFluid5MomentPlasmaEquationsEntropy2D)
    rho, rho_v1, rho_v2, rho_s = view(u, (4*i-3):(4*i))
    gamma = equations.gammas[i]
    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    p = rho^gamma * exp(rho_s / rho)
    if orientation == 1
        f1 = rho_v1
        f2 = rho_v1 * v1 + p
        f3 = rho_v1 * v2
        f4 = rho_s * v1
    else
        f1 = rho_v2
        f2 = rho_v2 * v1
        f3 = rho_v2 * v2 + p
        f4 = rho_s * v2
    end
    return SVector(f1, f2, f3, f4)
end

# Calculates the GLM-Maxwell flux at a single point
@inline function flux_glm_maxwell(u, orientation::Integer, equations::GlmMultiFluid5MomentPlasmaEquationsEntropy2D)
    if orientation == 1
        f1 = equations.c_e * equations.c_sqr * u[end]
        f2 = equations.c_sqr * u[end-1]
        f3 = u[end-2]
        f4 = equations.c_e * u[end-3]
    else
        f1 = -equations.c_sqr * u[end-1]
        f2 = equations.c_e * equations.c_sqr * u[end]
        f3 = -u[end-3]
        f4 = equations.c_e * u[end-2]
    end

    return SVector(f1, f2, f3, f4)
end

@inline function flux_central_upwind(u_ll, u_rr, orientation::Integer, equations::GlmMultiFluid5MomentPlasmaEquationsEntropy2D)
    fluxes_euler = SVector(ntuple(i -> flux_euler_central(u_ll, u_rr, orientation, i, equations), ncomponents(equations)))
    flux_glm = flux_glm_upwind(u_ll, u_rr, orientation, equations)
    return vcat(reduce(vcat, fluxes_euler), flux_glm)
end

@inline function flux_energy_central(u_ll, u_rr, orientation::Integer, equations::GlmMultiFluid5MomentPlasmaEquationsEntropy2D)
    prim_ll = cons2prim(u_ll, equations)
    prim_rr = cons2prim(u_rr, equations)
    fluxes_euler = SVector(ntuple(i -> flux_euler_energy_con(prim_ll, prim_rr, orientation, i, equations), ncomponents(equations)))
    flux_glm = flux_glm_central(u_ll, u_rr, orientation, equations)
    return vcat(reduce(vcat, fluxes_euler), flux_glm)
end

@inline function flux_energy_upwind(u_ll, u_rr, orientation::Integer, equations::GlmMultiFluid5MomentPlasmaEquationsEntropy2D)
    prim_ll = cons2prim(u_ll, equations)
    prim_rr = cons2prim(u_rr, equations)
    fluxes_euler = SVector(ntuple(i -> flux_euler_energy_con(prim_ll, prim_rr, orientation, i, equations), ncomponents(equations)))
    flux_glm = flux_glm_upwind(u_ll, u_rr, orientation, equations)
    return vcat(reduce(vcat, fluxes_euler), flux_glm)
end

@inline function flux_euler_central(u_ll, u_rr, orientation::Integer, i,
                              equations::GlmMultiFluid5MomentPlasmaEquationsEntropy2D)
    return 0.5f0 * (flux_euler(u_ll, orientation, i, equations) + flux_euler(u_rr, orientation, i, equations))
end

@inline function flux_glm_upwind(
    u_ll,
    u_rr,
    orientation::Integer,
    equations::GlmMultiFluid5MomentPlasmaEquationsEntropy2D,
)
    c = equations.speed_of_light
    c_e = equations.c_e
    n = 4*ncomponents(equations)
    u_sum = view(u_ll, (n+1):(n+4)) + view(u_rr, (n+1):(n+4))
    u_diff = view(u_ll, (n+1):(n+4)) - view(u_rr, (n+1):(n+4))
    if orientation == 1
        f1 = 0.5f0 * c * c_e * (u_diff[1] + c * u_sum[4])
        f2 = 0.5f0 * c * (u_diff[2] + c * u_sum[3])
        f3 = 0.5f0 * (u_sum[2] + c * u_diff[3])
        f4 = 0.5f0 * c_e * (u_sum[1] + c * u_diff[4])
    else
        f1 = 0.5f0 * c * (u_diff[1] - c * u_sum[3])
        f2 = 0.5f0 * c * c_e * (u_diff[2] + c * u_sum[4])
        f3 = 0.5f0 * (c * u_diff[3] - u_sum[1])
        f4 = 0.5f0 * c_e * (u_sum[2] + c * u_diff[4])
    end

    return SVector(f1, f2, f3, f4)
end

"""
    flux_euler_energy_con(u_ll, u_rr, orientation::Integer, i::Integer,
                          equations::GlmMultiFluid5MomentPlasmaEquationsEntropy2D)

"""
@inline function flux_euler_energy_con(prim_ll, prim_rr, orientation::Integer, i::Integer,
                              equations::GlmMultiFluid5MomentPlasmaEquationsEntropy2D)
    # Unpack left and right state
    rho_ll, v1_ll, v2_ll, p_ll = view(prim_ll, (4*i-3):(4*i))
    rho_rr, v1_rr, v2_rr, p_rr = view(prim_rr, (4*i-3):(4*i))
    gamma_m1 = equations.gammas[i] - 1
    gamma = equations.gammas[i]

    p_div_rho_ll = p_ll / rho_ll
    p_div_rho_rr = p_rr / rho_rr

    # Compute the necessary mean values
    rho_mean = ln_mean(rho_ll, rho_rr)
    ln_rho_avg = 0.5f0 * log(rho_ll * rho_rr)
    rho_avg = 0.5f0 * (rho_ll + rho_rr)
    v1_avg = 0.5f0 * (v1_ll + v1_rr)
    v2_avg = 0.5f0 * (v2_ll + v2_rr)
    p_avg = 0.5f0 * (p_ll + p_rr)
    p_div_rho_ln_ratio = Trixi.ln_ratio(p_div_rho_ll, p_div_rho_rr)

    # Calculate fluxes depending on orientation
    if orientation == 1
        f1 = rho_mean * v1_avg
        f2 = f1 * v1_avg + p_avg
        f3 = f1 * v2_avg
        f4 = f1 * (p_div_rho_ln_ratio - gamma - gamma_m1 * ln_rho_avg) +
             gamma_m1 * rho_avg * v1_avg
    else
        f1 = rho_mean * v2_avg
        f2 = f1 * v1_avg
        f3 = f1 * v2_avg + p_avg
        f4 = f1 * (p_div_rho_ln_ratio - gamma - gamma_m1 * ln_rho_avg) +
             gamma_m1 * rho_avg * v2_avg
    end

    return SVector(f1, f2, f3, f4)
end


@inline function flux_glm_central(u_ll, u_rr, orientation::Integer,
                              equations::GlmMultiFluid5MomentPlasmaEquationsEntropy2D)
    return 0.5f0 * (flux_glm_maxwell(u_ll, orientation, equations) + flux_glm_maxwell(u_rr, orientation, equations))
end

min_max_speed_naive(u_ll, u_rr, orientation, equations::GlmMultiFluid5MomentPlasmaEquationsEntropy2D) =
    max(1.0f0, equations.c_e) * (-equations.speed_of_light, equations.speed_of_light)

max_abs_speeds(u, equations::GlmMultiFluid5MomentPlasmaEquationsEntropy2D) =
    (max(1.0f0, equations.c_e) * equations.speed_of_light, max(1.0f0, equations.c_e) * equations.speed_of_light)

max_abs_speed_naive(u_ll, u_rr, orientation, equations::GlmMultiFluid5MomentPlasmaEquationsEntropy2D) =
    max(1.0f0, equations.c_e) * equations.speed_of_light


function source_term_lorentz(u, x, t, equations::GlmMultiFluid5MomentPlasmaEquationsEntropy2D)
    prim = cons2prim(u, equations)
    sources_euler = ntuple(i -> source_term_lorentz_euler(prim, x, t, i, equations), ncomponents(equations))
    sources_glm = source_term_lorentz_glm(u, x, t, equations)
    return vcat(sources_euler..., sources_glm)
end

function source_term_lorentz_corrected(u, x, t, equations::GlmMultiFluid5MomentPlasmaEquationsEntropy2D)
    prim = cons2prim(u, equations)
    sources_euler = ntuple(i -> source_term_lorentz_corrected_euler(prim, x, t, i, equations), ncomponents(equations))
    sources_glm = source_term_lorentz_glm(u, x, t, equations)
    return vcat(sources_euler..., sources_glm)
end

function source_term_lorentz_euler(prim, x, t, i, equations::GlmMultiFluid5MomentPlasmaEquationsEntropy2D)
    charge_mass_ratio = equations.charge_mass_ratios[i]
    gas_constant = equations.gas_constants[i]
    T_min = equations.T_min
    rho, v1, v2, p = view(prim, (4*i-3):(4*i))
    E1, E2, B, psi = prim[end-3], prim[end-2], prim[end-1], prim[end]

    s1 = 0
    s2 = charge_mass_ratio * rho * (E1 + v2 * B)
    s3 = charge_mass_ratio * rho * (E2 - v1 * B)
    s4 = 0
    return SVector(s1, s2, s3, s4)
end

function source_term_lorentz_corrected_euler(prim, x, t, i, equations::GlmMultiFluid5MomentPlasmaEquationsEntropy2D)
    charge_mass_ratio = equations.charge_mass_ratios[i]
    rho, v1, v2, p = view(prim, (4*i-3):(4*i))
    E1, E2, B, psi = prim[end-3], prim[end-2], prim[end-1], prim[end]
    gamma_m1 = equations.gammas[i] - 1

    s1 = 0
    s2 = charge_mass_ratio * rho * (E1 - B * v2)
    s3 = charge_mass_ratio * rho * (E2 + B * v1)
    s4 = -gamma_m1 * equations.c_sqr * equations.c_e * charge_mass_ratio * rho^2 * psi / p
    return SVector(s1, s2, s3, s4)
end

function source_term_lorentz_glm(u, x, t, equations::GlmMultiFluid5MomentPlasmaEquationsEntropy2D)
    current_density = Trixi.current_density(u, equations)
    charge_density = Trixi.charge_density(u, equations)
    inv_permittivity = inv(equations.permittivity)

    s1 = -current_density[1] * inv_permittivity
    s2 = -current_density[2] * inv_permittivity
    s3 = 0
    s4 = equations.c_e * inv_permittivity * charge_density
    
    return SVector(s1, s2, s3, s4)
end

@inline function density_pressure(u, equations::Trixi.GlmMultiFluid5MomentPlasmaEquationsEntropy2D)
    rhos = densities(u, equations)
    entr_s = entropies(u, equations)
    rho_times_p = rhos.^(equations.gammas .+ 1) .* exp.(entr_s ./ rhos)
    return minimum(rho_times_p)
end

#=
@inline function flux_upwind(
    u_ll,
    u_rr,
    orientation::Integer,
    equations::GLMMaxwellEquations2D,
)
    c = equations.speed_of_light
    c_h = equations.c_h
    u_sum = u_ll + u_rr
    u_diff = u_ll - u_rr
    if orientation == 1
        f1 = 0.5f0 * c * (c_h * u_diff[1] + c * u_sum[4])
        f2 = 0.5f0 * c * (u_diff[2] + c * u_sum[3])
        f3 = 0.5f0 * (u_sum[2] + c * u_diff[3])
        f4 = 0.5f0 * c_h * (c_h * u_sum[1] + c * u_diff[4])
    else
        f1 = 0.5f0 * c * (u_diff[1] - c * u_sum[3])
        f2 = 0.5f0 * c * (c_h * u_diff[2] + c * u_sum[4])
        f3 = 0.5f0 * (c * u_diff[3] - u_sum[1])
        f4 = 0.5f0 * c_h * (c_h * u_sum[2] + c * u_diff[4])
    end

    return SVector(f1, f2, f3, f4)
end


@inline function flux_upwind(
    u_ll,
    u_rr,
    normal_direction::AbstractVector,
    equations::GLMMaxwellEquations2D,
)
    c = equations.speed_of_light
    c_h = equations.c_h
    u_sum = u_ll + u_rr
    u_diff = u_ll - u_rr
    flux_component_1 =
        c_h *
        (normal_direction[1] * u_diff[1] + normal_direction[2] * u_diff[2]) +
        c * u_sum[4]
    flux_component_2 =
        normal_direction[1] * u_diff[2] - normal_direction[2] * u_diff[1] + c * u_sum[3]

    f1 =
        0.5f0 *
        c *
        (
            normal_direction[1] * flux_component_1 -
            normal_direction[2] * flux_component_2
        )
    f2 =
        0.5f0 *
        c *
        (
            normal_direction[2] * flux_component_1 +
            normal_direction[1] * flux_component_2
        )
    f3 =
        0.5f0 * (
            normal_direction[1] * u_sum[2] - normal_direction[2] * u_sum[1] +
            c * u_diff[3]
        )
    f4 =
        0.5f0 *
        c_h *
        (
            c_h *
            (normal_direction[1] * u_sum[1] + normal_direction[2] * u_sum[2]) +
            c * u_diff[4]
        )

    return SVector(f1, f2, f3, f4)
end

function boundary_condition_perfect_conducting_wall(
    u_inner,
    normal_direction::AbstractVector,
    direction,
    x,
    t,
    surface_flux_function,
    equations::GLMMaxwellEquations2D,
)
    psi_outer =
        2.0f0 *
        equations.c_h *
        (normal_direction[1] * u_inner[1] + normal_direction[2] * u_inner[2]) / equations.speed_of_light
    if iseven(direction)
        return surface_flux_function(
            u_inner,
            SVector(-u_inner[1], -u_inner[2], u_inner[3], -u_inner[4] - psi_outer),
            normal_direction,
            equations,
        )
    else
        return -surface_flux_function(
            u_inner,
            SVector(-u_inner[1], -u_inner[2], u_inner[3], -u_inner[4] + psi_outer),
            -normal_direction,
            equations,
        )
    end
end

function initial_condition_free_stream(x, t, equations::GLMMaxwellEquations2D)
    return SVector(10.0f0, 10.0f0, 10.0f0 / equations.speed_of_light, 10.0f0 / equations.speed_of_light)
end

function initial_condition_convergence_test(x, t, equations::GLMMaxwellEquations2D)
    c = equations.speed_of_light
    e1 = sin(x[2] + c * t)
    e2 = -sin(x[1] + c * t)
    b = (sin(x[1] + c * t) + sin(x[2] + c * t)) / c

    return SVector(e1, e2, b, 0.0f0)
end

min_max_speed_naive(u_ll, u_rr, orientation, equations::GLMMaxwellEquations2D) =
    max(1.0f0, equations.c_h) * (-equations.speed_of_light, equations.speed_of_light)

max_abs_speeds(u, equations::GLMMaxwellEquations2D) =
    (max(1.0f0, equations.c_h) * equations.speed_of_light, max(1.0f0, equations.c_h) * equations.speed_of_light)

max_abs_speed_naive(u_ll, u_rr, orientation, equations::GLMMaxwellEquations2D) =
    max(1.0f0, equations.c_h) * equations.speed_of_light
=#
end # @muladd
