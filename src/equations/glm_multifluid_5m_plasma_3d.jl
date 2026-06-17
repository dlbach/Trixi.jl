# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.

@muladd begin
#! format: noindent

struct GlmMultiFluid5MomentPlasmaEquations3D{NVARS, NCOMP, RealT <: Real} <: AbstractGlmMultiFluid5MomentPlasmaEquations{3, NVARS, NCOMP}
    gammas::SVector{NCOMP, RealT}               # specific heat ratio for each species
    inv_gammas_minus_one::SVector{NCOMP, RealT}  # = inv(gamma - 1); can be used to write slow divisions as fast multiplications
    gas_constants::SVector{NCOMP, RealT}        # specific gas constant for each species
    charge_mass_ratios::SVector{NCOMP, RealT}      # charge of one particle of each species divided by its mass
    speed_of_light::RealT                       # c
    c_sqr::RealT                                # squared speed of light
    permittivity::RealT                         # absolute dielectric permittivity
    permeability::RealT                         # magnetic permeability
    T_min::RealT                                # temperature bounding the fluid temperatures from below
    c_e::RealT                                  # GLM cleaning speed for the electric field
    function GlmMultiFluid5MomentPlasmaEquations3D{NVARS, NCOMP, RealT}(gammas::SVector{NCOMP, RealT},
                                                                    gas_constants::SVector{NCOMP, RealT},
                                                                    charge_mass_ratios::SVector{NCOMP, RealT},
                                                                    speed_of_light::RealT, permittivity::RealT,
                                                                    T_min::RealT, c_e::RealT) where {
                                                                                                                    NVARS,
                                                                                                                    NCOMP,
                                                                                                                    RealT <:
                                                                                                                    Real
                                                                                                                    }
        NCOMP >= 1 ||
            throw(DimensionMismatch("`gammas` and `charge_per_mass` have to be filled with at least one value"))

        inv_gammas_minus_one = inv.(gammas .- 1)
        permeability = inv(speed_of_light^2 * permittivity)
        new(gammas, inv_gammas_minus_one, gas_constants, charge_mass_ratios, speed_of_light, speed_of_light^2, permittivity, permeability, T_min, c_e)
    end
end

function GlmMultiFluid5MomentPlasmaEquations3D(gammas, gas_constants, charge_mass_ratios, speed_of_light = 299_792_458.0, 
                                            permittivity = 8.8541878188e-12, T_min = 1.0, c_e = 1.0)
    _gammas = promote(gammas...)
    _gas_constants = promote(gas_constants...)
    _charge_mass_ratios = promote(charge_mass_ratios...)

    RealT = promote_type(eltype(_gammas), eltype(_gas_constants), eltype(_charge_mass_ratios), typeof(speed_of_light), typeof(permittivity), typeof(T_min), typeof(c_e))
    
    _gammas = SVector(map(RealT, _gammas))
    _gas_constants = SVector(map(RealT, _gas_constants))
    _charge_mass_ratios = SVector(map(RealT, _charge_mass_ratios))

    speed_of_light = convert(RealT, speed_of_light)
    permittivity = convert(RealT, permittivity)
    T_min = convert(RealT, T_min)
    c_e = convert(RealT, c_e)

    NVARS = 5*length(_gammas) + 8
    NCOMP = length(_gammas)
    return GlmMultiFluid5MomentPlasmaEquations3D{NVARS, NCOMP, RealT}(_gammas, _gas_constants, _charge_mass_ratios, speed_of_light, permittivity, T_min, c_e)
end
    
function varnames(::typeof(cons2cons), equations::GlmMultiFluid5MomentPlasmaEquations3D)
    fluids = ntuple(n -> SVector("rho_" * string(n), "rho_v1_" * string(n), "rho_v2_" * string(n), "rho_v3_" * string(n), "rho_e_total_" * string(n)), Val(ncomponents(equations)))
    glm = ("E1", "E2", "E3", "B1", "B2", "B3", "psi_E", "psi_B")
    return (reduce(vcat, fluids)..., glm...)
end

@inline function Base.real(::GlmMultiFluid5MomentPlasmaEquations3D{NVARS, NCOMP, RealT}) where {
                                                                                               NVARS,
                                                                                               NCOMP,
                                                                                               RealT
                                                                                               }
    return RealT
end

# Convert conservative vaiables to primitive
@inline function cons2prim(u, equations::GlmMultiFluid5MomentPlasmaEquations3D)
    prims_euler = SVector(ntuple(i -> cons2prim_euler(u, i, equations), ncomponents(equations)))
    prims_glm = SVector(u[end-3], u[end-2], u[end-1], u[end])
    return vcat(reduce(vcat, prims_euler), prims_glm)
end

@inline function cons2prim_euler(u, i, equations::GlmMultiFluid5MomentPlasmaEquations3D)
    rho, rho_v1, rho_v2, rho_v3, rho_e = view(u, (8*i-7):(8*i))

    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    v3 = rho_v3 / rho
    p = (equations.gammas[i] - 1) * (rho_e - 0.5f0 * (rho_v1 * v1 + rho_v2 * v2 + rho_v3 * v3))

    return SVector(rho, v1, v2, v3, p)
end

@inline function prim2cons(prim, equations::GlmMultiFluid5MomentPlasmaEquations3D)
    prims_euler = SVector(ntuple(i -> prim2cons_euler(prim, i, equations), ncomponents(equations)))
    prims_glm = SVector(prim[end-7], prim[end-6], prim[end-5], prim[end-4], prim[end-3], prim[end-2], prim[end-1], prim[end])
    return vcat(reduce(vcat, prims_euler), prims_glm)
end

# Convert primitive to conservative variables
@inline function prim2cons_euler(prim, i, equations::GlmMultiFluid5MomentPlasmaEquations3D)
    rho, v1, v2, v3, p = view(u, (8*i-7):(8*i))
    rho_v1 = rho * v1
    rho_v2 = rho * v2
    rho_v3 = rho * v3
    rho_e = p * equations.inv_gammas_minus_one[i] + 0.5f0 * (rho_v1 * v1 + rho_v2 * v2 + rho_v3 * v3)
    return SVector(rho, rho_v1, rho_v2, rho_e)
end

# Convert conservative variables to entropy variables
@inline function cons2entropy(u, equations::GlmMultiFluid5MomentPlasmaEquations3D)
    entropy_euler = SVector(ntuple(i -> cons2entropy_euler(u, i, equations), ncomponents(equations)))
    entropy_glm = SVector(u[end-3]*equations.permittivity, u[end-2]*equations.permittivity, 
                        u[end-1]/equations.permeability, u[end]/equations.permeability) / equations.T_min
    return vcat(reduce(vcat, entropy_euler), entropy_glm)
end

@inline function cons2entropy_euler(u, i, equations::GlmMultiFluid5MomentPlasmaEquations3D)
    rho, rho_v1, rho_v2, rho_e = view(u, (4*i-3):(4*i))
    gas_constant = equations.gas_constants[i]

    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    v_square = v1^2 + v2^2
    p = (equations.gammas[i] - 1) * (rho_e - 0.5f0 * rho * v_square)
    s = log(p) - equations.gammas[i] * log(rho)
    rho_p = rho / p

    w1 = gas_constant * ((equations.gammas[i] - s) * equations.inv_gammas_minus_one[i] -
         0.5f0 * rho_p * v_square)
    w2 = gas_constant * rho_p * v1
    w3 = gas_constant * rho_p * v2
    w4 = inv(equations.T_min) - gas_constant * rho_p

    return SVector(w1, w2, w3, w4)
end

# Convert entropy variables to conservative variables
@inline function entropy2cons(w, equations::GlmMultiFluid5MomentPlasmaEquations3D)
    cons_euler = SVector(ntuple(i -> entropy2cons_euler(w, i, equations), ncomponents(equations)))
    cons_glm = equations.T_min * SVector(w[end-3]/equations.permittivity, w[end-2]/equations.permittivity, 
                                        w[end-1]*equations.permeability, w[end]*equations.permeability)
    return vcat(reduce(vcat, cons_euler), cons_glm)
end

@inline function entropy2cons_euler(w, i, equations::GlmMultiFluid5MomentPlasmaEquations3D)
    # See Hughes, Franca, Mallet (1986) A new finite element formulation for CFD
    # [DOI: 10.1016/0045-7825(86)90127-1](https://doi.org/10.1016/0045-7825(86)90127-1)
    gamma = equations.gammas[i]
    inv_gamma_minus_one = equations.inv_gammas_minus_one[i]
    # convert to entropy `-rho * s` used by Hughes, France, Mallet (1986)
    # instead of `-rho * s / (gamma - 1)`
    V1, V2, V3, V5 = view(w, (4*i-3):(4*i)) .* (gamma - 1)/equations.gas_constants[i]
    V5 -= (gamma - 1)/(equations.T_min * equations.gas_constants[i])

    # s = specific entropy, eq. (53)
    s = gamma - V1 + (V2^2 + V3^2) / (2 * V5)

    # eq. (52)
    rho_iota = ((gamma - 1) / (-V5)^gamma)^(inv_gamma_minus_one) *
               exp(-s * inv_gamma_minus_one)

    # eq. (51)
    rho = -rho_iota * V5
    rho_v1 = rho_iota * V2
    rho_v2 = rho_iota * V3
    rho_e = rho_iota * (1 - (V2^2 + V3^2) / (2 * V5))
    return SVector(rho, rho_v1, rho_v2, rho_e)
end

function default_analysis_integrals(::GlmMultiFluid5MomentPlasmaEquations3D)
    (Val(:l2_dive), Val(:l2_e_normal_jump), entropy_timederivative)
end

@inline electric_field(u, equations::GlmMultiFluid5MomentPlasmaEquations3D) = SVector(u[end-3], u[end-2])

@inline function charge_density(u, equations::GlmMultiFluid5MomentPlasmaEquations3D) 
    return sum(densities(u, equations)[i] * equations.charge_mass_ratios[i] for i in 1:ncomponents(equations))
end

@inline function current_density(u, equations::GlmMultiFluid5MomentPlasmaEquations3D) 
    return sum(momenta(u, equations)[i] * equations.charge_mass_ratios[i] for i in 1:ncomponents(equations))
end

@inline function scaled_charge_density(u, x, t, source_terms, equations::GlmMultiFluid5MomentPlasmaEquations3D) 
    return charge_density(u, equations) / equations.permittivity
end

@inline function densities(u, equations::GlmMultiFluid5MomentPlasmaEquations3D)
    return SVector(ntuple(i -> u[4*i-3], ncomponents(equations)))
end

@inline function momenta(u, equations::GlmMultiFluid5MomentPlasmaEquations3D)
    return SVector(ntuple(i -> SVector(u[4*i-2], u[4*i-1]), ncomponents(equations)))
end


@inline function flux(u, orientation::Integer, equations::GlmMultiFluid5MomentPlasmaEquations3D)
    fluxes_euler = ntuple(i -> flux_euler(u, orientation, i, equations), ncomponents(equations))
    flux_glm = flux_glm_maxwell(u, orientation, equations)
    return vcat(fluxes_euler..., flux_glm)
end

# Calculates the Euler flux for a single species at a single point
@inline function flux_euler(u, orientation::Integer, i::Integer, equations::GlmMultiFluid5MomentPlasmaEquations3D)
    rho, rho_v1, rho_v2, rho_e = view(u, (4*i-3):(4*i))
    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    p = (equations.gammas[i] - 1) * (rho_e - 0.5f0 * (rho_v1 * v1 + rho_v2 * v2))
    if orientation == 1
        f1 = rho_v1
        f2 = rho_v1 * v1 + p
        f3 = rho_v1 * v2
        f4 = (rho_e + p) * v1
    else
        f1 = rho_v2
        f2 = rho_v2 * v1
        f3 = rho_v2 * v2 + p
        f4 = (rho_e + p) * v2
    end
    return SVector(f1, f2, f3, f4)
end

# Calculates the GLM-Maxwell flux at a single point
@inline function flux_glm_maxwell(u, orientation::Integer, equations::GlmMultiFluid5MomentPlasmaEquations3D)
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

@inline function flux_central_upwind(u_ll, u_rr, orientation::Integer, equations::GlmMultiFluid5MomentPlasmaEquations3D)
    fluxes_euler = SVector(ntuple(i -> flux_euler_central(u_ll, u_rr, orientation, i, equations), ncomponents(equations)))
    flux_glm = flux_glm_upwind(u_ll, u_rr, orientation, equations)
    return vcat(reduce(vcat, fluxes_euler), flux_glm)
end

@inline function flux_ranocha_central(u_ll, u_rr, orientation::Integer, equations::GlmMultiFluid5MomentPlasmaEquations3D)
    prim_ll = cons2prim(u_ll, equations)
    prim_rr = cons2prim(u_rr, equations)
    fluxes_euler = SVector(ntuple(i -> flux_euler_ranocha(prim_ll, prim_rr, orientation, i, equations), ncomponents(equations)))
    flux_glm = flux_glm_central(u_ll, u_rr, orientation, equations)
    return vcat(reduce(vcat, fluxes_euler), flux_glm)
end

@inline function flux_euler_central(u_ll, u_rr, orientation::Integer, i,
                              equations::GlmMultiFluid5MomentPlasmaEquations3D)
    return 0.5f0 * (flux_euler(u_ll, orientation, i, equations) + flux_euler(u_rr, orientation, i, equations))
end

@inline function flux_glm_upwind(
    u_ll,
    u_rr,
    orientation::Integer,
    equations::GlmMultiFluid5MomentPlasmaEquations3D,
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
    flux_euler_ranocha(u_ll, u_rr, orientation::Integer, i::Integer,
                          equations::GlmMultiFluid5MomentPlasmaEquations3D)

Entropy conserving and kinetic energy preserving two-point flux by
- Hendrik Ranocha (2018)
  Generalised Summation-by-Parts Operators and Entropy Stability of Numerical Methods
  for Hyperbolic Balance Laws
  [PhD thesis, TU Braunschweig](https://cuvillier.de/en/shop/publications/7743)
See also
- Hendrik Ranocha (2020)
  Entropy Conserving and Kinetic Energy Preserving Numerical Methods for
  the Euler Equations Using Summation-by-Parts Operators
  [Proceedings of ICOSAHOM 2018](https://doi.org/10.1007/978-3-030-39647-3_42)
"""
@inline function flux_euler_ranocha(prim_ll, prim_rr, orientation::Integer, i::Integer,
                              equations::GlmMultiFluid5MomentPlasmaEquations3D)
    # Unpack left and right state
    rho_ll, v1_ll, v2_ll, p_ll = view(prim_ll, (4*i-3):(4*i))
    rho_rr, v1_rr, v2_rr, p_rr = view(prim_rr, (4*i-3):(4*i))

    # Compute the necessary mean values
    rho_mean = ln_mean(rho_ll, rho_rr)
    # Algebraically equivalent to `inv_ln_mean(rho_ll / p_ll, rho_rr / p_rr)`
    # in exact arithmetic since
    #     log((ϱₗ/pₗ) / (ϱᵣ/pᵣ)) / (ϱₗ/pₗ - ϱᵣ/pᵣ)
    #   = pₗ pᵣ log((ϱₗ pᵣ) / (ϱᵣ pₗ)) / (ϱₗ pᵣ - ϱᵣ pₗ)
    inv_rho_p_mean = p_ll * p_rr * inv_ln_mean(rho_ll * p_rr, rho_rr * p_ll)
    v1_avg = 0.5f0 * (v1_ll + v1_rr)
    v2_avg = 0.5f0 * (v2_ll + v2_rr)
    p_avg = 0.5f0 * (p_ll + p_rr)
    velocity_square_avg = 0.5f0 * (v1_ll * v1_rr + v2_ll * v2_rr)

    # Calculate fluxes depending on orientation
    if orientation == 1
        f1 = rho_mean * v1_avg
        f2 = f1 * v1_avg + p_avg
        f3 = f1 * v2_avg
        f4 = f1 *
             (velocity_square_avg + inv_rho_p_mean * equations.inv_gammas_minus_one[i]) +
             0.5f0 * (p_ll * v1_rr + p_rr * v1_ll)
    else
        f1 = rho_mean * v2_avg
        f2 = f1 * v1_avg
        f3 = f1 * v2_avg + p_avg
        f4 = f1 *
             (velocity_square_avg + inv_rho_p_mean * equations.inv_gammas_minus_one[i]) +
             0.5f0 * (p_ll * v2_rr + p_rr * v2_ll)
    end

    return SVector(f1, f2, f3, f4)
end


@inline function flux_glm_central(u_ll, u_rr, orientation::Integer,
                              equations::GlmMultiFluid5MomentPlasmaEquations3D)
    return 0.5f0 * (flux_glm_maxwell(u_ll, orientation, equations) + flux_glm_maxwell(u_rr, orientation, equations))
end

min_max_speed_naive(u_ll, u_rr, orientation, equations::GlmMultiFluid5MomentPlasmaEquations3D) =
    max(1.0f0, equations.c_e) * (-equations.speed_of_light, equations.speed_of_light)

max_abs_speeds(u, equations::GlmMultiFluid5MomentPlasmaEquations3D) =
    (max(1.0f0, equations.c_e) * equations.speed_of_light, max(1.0f0, equations.c_e) * equations.speed_of_light)

max_abs_speed_naive(u_ll, u_rr, orientation, equations::GlmMultiFluid5MomentPlasmaEquations3D) =
    max(1.0f0, equations.c_e) * equations.speed_of_light


function source_term_lorentz_corrected(u, x, t, equations::GlmMultiFluid5MomentPlasmaEquations3D)
    prim = cons2prim(u, equations)
    sources_euler = ntuple(i -> source_term_lorentz_euler(prim, x, t, i, equations), ncomponents(equations))
    sources_glm = source_term_lorentz_glm(u, x, t, equations)
    return vcat(sources_euler..., sources_glm)
end

function source_term_lorentz_euler(prim, x, t, i, equations::GlmMultiFluid5MomentPlasmaEquations3D)
    charge_mass_ratio = equations.charge_mass_ratios[i]
    gas_constant = equations.gas_constants[i]
    T_min = equations.T_min
    rho, v1, v2, p = view(prim, (4*i-3):(4*i))
    E1, E2, B, psi = prim[end-3], prim[end-2], prim[end-1], prim[end]

    s1 = 0.0f0
    s2 = charge_mass_ratio * rho * (E1 - B * v2)
    s3 = charge_mass_ratio * rho * (E2 + B * v1)
    s4 = charge_mass_ratio * rho * ( (v1 * E1 + v2 * E2) - (equations.c_e * equations.c_sqr * psi)/(1 - gas_constant * T_min * (rho/p)) )
    return SVector(s1, s2, s3, s4)
end

function source_term_lorentz_glm(u, x, t, equations::GlmMultiFluid5MomentPlasmaEquations3D)
    current_density = Trixi.current_density(u, equations)
    charge_density = Trixi.charge_density(u, equations)
    inv_permittivity = inv(equations.permittivity)

    s1 = -current_density[1] * inv_permittivity
    s2 = -current_density[2] * inv_permittivity
    s3 = 0.0f0
    s4 = equations.c_e * inv_permittivity * charge_density
    
    return SVector(s1, s2, s3, s4)
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
