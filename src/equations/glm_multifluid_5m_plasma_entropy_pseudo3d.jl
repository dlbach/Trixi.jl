# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.

@muladd begin
#! format: noindent

struct GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D{NVARS, NCOMP, RealT <: Real} <: AbstractGlmMultiFluid5MomentPlasmaEquationsEntropy{2, NVARS, NCOMP}
    gammas::SVector{NCOMP, RealT}               # specific heat ratio for each species
    inv_gammas_minus_one::SVector{NCOMP, RealT}  # = inv(gamma - 1); can be used to write slow divisions as fast multiplications
    gas_constants::SVector{NCOMP, RealT}        # specific gas constant for each species
    charge_mass_ratios::SVector{NCOMP, RealT}      # charge of one particle of each species divided by its mass
    speed_of_light::RealT                       # c
    c_sqr::RealT                                # squared speed of light
    permittivity::RealT                         # absolute dielectric permittivity
    permeability::RealT                         # magnetic permeability
    c_e::RealT                                  # GLM cleaning speed for the electric field
    c_b::RealT                                  # GLM cleaning speed for the magnetic field
    function GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D{NVARS, NCOMP, RealT}(gammas::SVector{NCOMP, RealT},
                                                                    gas_constants::SVector{NCOMP, RealT},
                                                                    charge_mass_ratios::SVector{NCOMP, RealT},
                                                                    speed_of_light::RealT, permittivity::RealT,
                                                                    c_e::RealT, c_b::RealT) where {
                                                                                                                    NVARS,
                                                                                                                    NCOMP,
                                                                                                                    RealT <:
                                                                                                                    Real
                                                                                                                    }
        NCOMP >= 1 ||
            throw(DimensionMismatch("`gammas` and `charge_per_mass` have to be filled with at least one value"))

        inv_gammas_minus_one = inv.(gammas .- 1)
        permeability = inv(speed_of_light^2 * permittivity)
        new(gammas, inv_gammas_minus_one, gas_constants, charge_mass_ratios, speed_of_light, speed_of_light^2, permittivity, permeability, c_e, c_b)
    end
end

function GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D(gammas, gas_constants, charge_mass_ratios, speed_of_light = 299_792_458.0, 
                                            permittivity = 8.8541878188e-12, c_e = 1.0, c_b = 1.0)
    _gammas = promote(gammas...)
    _gas_constants = promote(gas_constants...)
    _charge_mass_ratios = promote(charge_mass_ratios...)

    RealT = promote_type(eltype(_gammas), eltype(_gas_constants), eltype(_charge_mass_ratios), typeof(speed_of_light), typeof(permittivity), typeof(c_e), typeof(c_b))
    
    _gammas = SVector(map(RealT, _gammas))
    _gas_constants = SVector(map(RealT, _gas_constants))
    _charge_mass_ratios = SVector(map(RealT, _charge_mass_ratios))

    speed_of_light = convert(RealT, speed_of_light)
    permittivity = convert(RealT, permittivity)
    c_e = convert(RealT, c_e)
    c_b = convert(RealT, c_b)

    NVARS = 5*length(_gammas) + 8
    NCOMP = length(_gammas)
    return GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D{NVARS, NCOMP, RealT}(_gammas, _gas_constants, _charge_mass_ratios, speed_of_light, permittivity, c_e, c_b)
end
    
function varnames(::typeof(cons2cons), equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D)
    fluids = ntuple(n -> SVector("rho_" * string(n), "rho_v1_" * string(n), "rho_v2_" * string(n), "rho_v3_" * string(n), "rho_s_" * string(n)), Val(ncomponents(equations)))
    glm = ("E1", "E2", "E3", "B1", "B2", "B3", "psi_E", "psi_B")
    return (reduce(vcat, fluids)..., glm...)
end

function varnames(::typeof(cons2prim), equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D)
    fluids = ntuple(n -> SVector("rho_" * string(n), "v1_" * string(n), "v2_" * string(n), "v3_" * string(n), "p_" * string(n)), Val(ncomponents(equations)))
    glm = ("E1", "E2", "E3", "B1", "B2", "B3", "psi_E", "psi_B")
    return (reduce(vcat, fluids)..., glm...)
end

@inline function Base.real(::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D{NVARS, NCOMP, RealT}) where {
                                                                                               NVARS,
                                                                                               NCOMP,
                                                                                               RealT
                                                                                               }
    return RealT
end

have_nonconservative_terms(::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D) = Trixi.True()

# Convert conservative vaiables to primitive
@inline function cons2prim(u, equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D)
    prims_euler = SVector(ntuple(i -> cons2prim_euler(u, i, equations), ncomponents(equations)))
    prims_glm = SVector(u[end-7], u[end-6], u[end-5], u[end-4], u[end-3], u[end-2], u[end-1], u[end])
    return vcat(reduce(vcat, prims_euler), prims_glm)
end

@inline function cons2prim_euler(u, i, equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D)
    rho, rho_v1, rho_v2, rho_v3, rho_s = view(u, (5*i-4):(5*i))
    gamma = equations.gammas[i]

    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    v3 = rho_v3 / rho
    s = rho_s / rho
    p = rho^gamma * exp(s)

    return SVector(rho, v1, v2, v3, p)
end

@inline function prim2cons(prim, equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D)
    prims_euler = SVector(ntuple(i -> prim2cons_euler(prim, i, equations), ncomponents(equations)))
    prims_glm = SVector(prim[end-7], prim[end-6], prim[end-5], prim[end-4], prim[end-3], prim[end-2], prim[end-1], prim[end])
    return vcat(reduce(vcat, prims_euler), prims_glm)
end

# Convert primitive to conservative variables
@inline function prim2cons_euler(prim, i, equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D)
    rho, v1, v2, v3, p = view(prim, (5*i-4):(5*i))
    gamma = equations.gammas[i]

    rho_v1 = rho * v1
    rho_v2 = rho * v2
    rho_v3 = rho * v3
    rho_s = rho * log(p / rho^gamma)

    return SVector(rho, rho_v1, rho_v2, rho_v3, rho_s)
end

# Convert conservative variables to entropy variables
@inline function cons2entropy(u, equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D)
    entropy_euler = SVector(ntuple(i -> cons2entropy_euler(u, i, equations), ncomponents(equations)))
    entropy_glm = SVector(u[end-7]*equations.permittivity, u[end-6]*equations.permittivity, 
                          u[end-5]*equations.permittivity, u[end-4]/equations.permeability, 
                          u[end-3]/equations.permeability, u[end-2]/equations.permeability,
                          u[end-1]/equations.permeability, u[end]*equations.permittivity)
    return vcat(reduce(vcat, entropy_euler), entropy_glm)
end

@inline function cons2entropy_euler(u, i, equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D)
    rho, rho_v1, rho_v2, rho_v3, rho_s = view(u, (5*i-4):(5*i))
    gamma = equations.gammas[i]
    inv_gamma_minus_one = equations.inv_gammas_minus_one[i]

    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    v3 = rho_v3 / rho
    s = rho_s / rho
    p = rho^(gamma) * exp(s)
    v_square = v1^2 + v2^2 + v3^2
    p_div_rho = p / rho

    w = -0.5f0 * v_square + inv_gamma_minus_one * p_div_rho * (gamma - s)

    return SVector(w, v1, v2, v3, inv_gamma_minus_one * p_div_rho)
end

@inline function cons2entropy_euler_classic(u, equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D)
    rho, rho_v1, rho_v2, rho_v3, rho_s = u

    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    v3 = rho_v3 / rho
    v_square = v1^2 + v2^2 + v3^2
    s = rho_s / rho
    p = rho^(equations.gammas[1]) * exp(s)
    rho_p = rho / p

    w1 = (equations.gammas[1] - s) * equations.inv_gammas_minus_one[1] -
         0.5f0 * rho_p * v_square
    w2 = rho_p * v1
    w3 = rho_p * v2
    w4 = rho_p * v3
    w5 = -rho_p

    return (equations.gammas[1] - 1) * SVector(w1, w2, w3, w4, w5)
end

# Convert entropy variables to conservative variables
@inline function entropy2cons(w, equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D)
    cons_euler = SVector(ntuple(i -> entropy2cons_euler(w, i, equations), ncomponents(equations)))
    cons_glm = SVector(w[end-7]/equations.permittivity, w[end-6]/equations.permittivity, 
                       w[end-5]/equations.permittivity, w[end-4]*equations.permeability, 
                       w[end-3]*equations.permeability, w[end-2]*equations.permeability,
                       w[end-1]*equations.permeability, w[end]/equations.permittivity)
    return vcat(reduce(vcat, cons_euler), cons_glm)
end

@inline function entropy2cons_euler(w, i, equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D)
    gamma = equations.gammas[i]
    inv_gamma_minus_one = equations.inv_gammas_minus_one[i]
    V1, V2, V3, V4, V5 = view(w, (5*i-4):(5*i))
    
    v_square = V2^2 + V3^2 + V4^2
    p_div_rho =  (gamma - 1) * V5
    s = (V1 + 0.5f0 * v_square - gamma * rho_p / inv_gamma_minus_one) * (gamma - 1) / p_div_rho
    rho_gamma_minus_one = inv(exp(s) / p_div_rho)

    rho = exp(log(rho_gamma_minus_one) * inv_gamma_minus_one)
    rho_v1 = rho * V2
    rho_v2 = rho * V3
    rho_v3 = rhp * V4
    rho_s = rho * s
    return SVector(rho, rho_v1, rho_v2, rho_v3, rho_s)
end

function default_analysis_integrals(::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D)
    (Val(:l2_dive), Val(:l2_e_normal_jump), Val(:l2_divb), Val(:l2_b_normal_jump), entropy_timederivative)
end

@inline electric_field(u, equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D) = SVector(u[end-7], u[end-6], u[end-5])

@inline magnetic_field(u, equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D) = SVector(u[end-4], u[end-3], u[end-2])

@inline function charge_density(u, equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D) 
    rhos = densities(u, equations)
    return sum(rhos[i] * equations.charge_mass_ratios[i] for i in 1:ncomponents(equations))
end

@inline function current_density(u, equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D) 
    return sum(momenta(u, equations)[i] * equations.charge_mass_ratios[i] for i in 1:ncomponents(equations))
end

@inline function scaled_charge_density(u, x, t, source_terms, equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D) 
    return charge_density(u, equations) / equations.permittivity
end

@inline function densities(u, equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D)
    return SVector(ntuple(i -> u[5*i-4], ncomponents(equations)))
end

@inline function momenta(u, equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D)
    return SVector(ntuple(i -> SVector(u[5*i-3], u[5*i-2], u[5*i-1]), ncomponents(equations)))
end

@inline function entropies(u, equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D)
    return SVector(ntuple(i -> u[5*i], ncomponents(equations)))
end

@inline function flux(u, orientation_or_normal_direction, equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D)
    fluxes_euler = ntuple(i -> flux_euler(u, orientation_or_normal_direction, i, equations), ncomponents(equations))
    flux_glm = flux_glm_maxwell(u, orientation_or_normal_direction, equations)
    return vcat(fluxes_euler..., flux_glm)
end

# Calculates the Euler flux for a single species at a single point
@inline function flux_euler(u, orientation::Integer, i::Integer, equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D)
    rho, rho_v1, rho_v2, rho_v3, rho_s = view(u, (5*i-4):(5*i))
    gamma = equations.gammas[i]
    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    v3 = rho_v3 / rho
    p = rho^gamma * exp(rho_s / rho)
    if orientation == 1
        f1 = rho_v1
        f2 = rho_v1 * v1 + p
        f3 = rho_v1 * v2
        f4 = rho_v1 * v3
        f5 = rho_s * v1
    else
        f1 = rho_v2
        f2 = rho_v2 * v1
        f3 = rho_v2 * v2 + p
        f4 = rho_v2 * v3
        f5 = rho_s * v2
    end
    return SVector(f1, f2, f3, f4, f5)
end

# Calculate 1D flux for a single point in the normal direction
# Note, this directional vector is not normalized
@inline function flux_euler(u, normal_direction::AbstractVector, i::Integer, equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D)
    rho, rho_v1, rho_v2, rho_v3, rho_s = view(u, (5*i-4):(5*i))
    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    v3 = rho_v3 / rho
    p = rho^equations.gammas[i] * exp(rho_s / rho)

    v_normal = v1 * normal_direction[1] + v2 * normal_direction[2]
    rho_v_normal = rho * v_normal

    f1 = rho_v_normal
    f2 = rho_v_normal * v1 + p * normal_direction[1]
    f3 = rho_v_normal * v2 + p * normal_direction[2]
    f4 = rho_v_normal * v3
    f5 = rho_s * v_normal

    return SVector(f1, f2, f3, f4, f5)
end

# Calculates the GLM-Maxwell flux at a single point
@inline function flux_glm_maxwell(u, orientation::Integer, equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D)
    c_sqr = equations.c_sqr
    c_e = equations.c_e
    c_b = equations.c_b

    if orientation == 1
        f1 = c_e * c_sqr * u[end-1]
        f2 = c_sqr * u[end-2]
        f3 = -c_sqr * u[end-3]
        f4 = c_b * u[end]
        f5 = -u[end-5]
        f6 = u[end-6]
        f7 = c_e * u[end-7]
        f8 = c_b * c_sqr * u[end-4]
    else
        f1 = -c_sqr * u[end-2]
        f2 = c_e * c_sqr * u[end-1]
        f3 = c_sqr * u[end-4]
        f4 = u[end-5]
        f5 = c_b * u[end]
        f6 = -u[end-7]
        f7 = c_e * u[end-6]
        f8 = c_b * c_sqr * u[end-3]
    end

    return SVector(f1, f2, f3, f4, f5, f6, f7, f8)
end

#=
# Calculates the GLM-Maxwell flux at a single point
@inline function flux_glm_maxwell(u, normal_direction::AbstractVector, equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D)
    c_sqr = equations.c_sqr
    c_e = equations.c_e
    c_b = equations.c_b

    f1 = c_sqr * (c_e * u[end-1] * normal_direction[1] - u[end-2] * normal_direction[2] + u[end-3] * normal_direction[3])
    f2 = c_sqr * (u[end-2] * normal_direction[1] + c_e * u[end-1] * normal_direction[2] - u[end-4] * normal_direction[3])
    f3 = c_sqr * (-u[end-3] * normal_direction[1] + u[end-4] * normal_direction[2] + c_e * u[end-1] * normal_direction[3])
    f4 = c_b * u[end] * normal_direction[1] + u[end-5] * normal_direction[2] - u[end-6] * normal_direction[3]
    f5 = -u[end-5] * normal_direction[1] + c_b * u[end] * normal_direction[2] + u[end-7] * normal_direction[3]
    f6 = u[end-6] * normal_direction[1] - u[end-7] * normal_direction[2] + c_b * u[end] * normal_direction[3]
    f7 = c_e * (u[end-7] * normal_direction[1] + u[end-6] * normal_direction[2] + u[end-5] * normal_direction[3])
    f8 = c_b * c_sqr * (u[end-4] * normal_direction[1] + u[end-3] * normal_direction[2] + u[end-2] * normal_direction[3])

    return SVector(f1, f2, f3, f4, f5, f6, f7, f8)
end
=#
@inline function flux_central_upwind(u_ll, u_rr, orientation_or_normal_direction, equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D)
    fluxes_euler = SVector(ntuple(i -> flux_euler_central(u_ll, u_rr, orientation_or_normal_direction, i, equations), ncomponents(equations)))
    flux_glm = flux_glm_upwind(u_ll, u_rr, orientation_or_normal_direction, equations)
    return vcat(reduce(vcat, fluxes_euler), flux_glm)
end

@inline function flux_energy_central(u_ll, u_rr, orientation_or_normal_direction, equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D)
    prim_ll = cons2prim(u_ll, equations)
    prim_rr = cons2prim(u_rr, equations)
    fluxes_euler = SVector(ntuple(i -> flux_euler_energy_con(prim_ll, prim_rr, orientation_or_normal_direction, i, equations), ncomponents(equations)))
    flux_glm = flux_glm_central(u_ll, u_rr, orientation_or_normal_direction, equations)
    return vcat(reduce(vcat, fluxes_euler), flux_glm)
end

@inline function flux_energy_upwind(u_ll, u_rr, orientation_or_normal_direction, equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D)
    prim_ll = cons2prim(u_ll, equations)
    prim_rr = cons2prim(u_rr, equations)
    fluxes_euler = SVector(ntuple(i -> flux_euler_energy_con(prim_ll, prim_rr, orientation_or_normal_direction, i, equations), ncomponents(equations)))
    flux_glm = flux_glm_upwind(u_ll, u_rr, orientation_or_normal_direction, equations)
    return vcat(reduce(vcat, fluxes_euler), flux_glm)
end

@inline function flux_energy_diss_upwind(u_ll, u_rr, orientation_or_normal_direction, equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D)
    prim_ll = cons2prim(u_ll, equations)
    prim_rr = cons2prim(u_rr, equations)
    fluxes_euler = SVector(ntuple(i -> flux_euler_energy_diss(prim_ll, prim_rr, orientation_or_normal_direction, i, equations), ncomponents(equations)))
    flux_glm = flux_glm_upwind(u_ll, u_rr, orientation_or_normal_direction, equations)
    return vcat(reduce(vcat, fluxes_euler), SVector{typeof(equations.gammas[1])})
end

@inline function flux_euler_central(u_ll, u_rr, orientation_or_normal_direction, i, equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D)
    return 0.5f0 * (flux_euler(u_ll, orientation_or_normal_direction, i, equations) + flux_euler(u_rr, orientation_or_normal_direction, i, equations))
end

@inline function (dissipation::DissipationMatrixWintersEtal)(u_ll, u_rr,
                                                             normal_direction::AbstractVector,
                                                             equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D)
    prim_ll = cons2prim(u_ll, equations)
    prim_rr = cons2prim(u_rr, equations)
    fluxes_euler = SVector(ntuple(i -> flux_euler_noncon_dissipation_winters_etal(prim_ll, prim_rr, normal_direction, i, equations), ncomponents(equations)))
    return vcat(reduce(vcat, fluxes_euler), zeros(SVector{8, typeof(equations.gammas[1])}))
end
    

"""
    flux_euler_energy_con(u_ll, u_rr, orientation::Integer, i::Integer,
                          equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D)

"""
@inline function flux_euler_energy_con(prim_ll, prim_rr, orientation::Integer, i::Integer,
                              equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D)
    # Unpack left and right state
    rho_ll, v1_ll, v2_ll, v3_ll, p_ll = view(prim_ll, (5*i-4):(5*i))
    rho_rr, v1_rr, v2_rr, v3_rr, p_rr = view(prim_rr, (5*i-4):(5*i))
    gamma = equations.gammas[i]
    gamma_m1 = gamma - 1

    p_div_rho_ll = p_ll / rho_ll
    p_div_rho_rr = p_rr / rho_rr

    # Compute the necessary mean values
    rho_mean = ln_mean(rho_ll, rho_rr)
    ln_rho_avg = 0.5f0 * log(rho_ll * rho_rr)
    rho_avg = 0.5f0 * (rho_ll + rho_rr)
    v1_avg = 0.5f0 * (v1_ll + v1_rr)
    v2_avg = 0.5f0 * (v2_ll + v2_rr)
    v3_avg = 0.5f0 * (v3_ll + v3_rr)
    p_avg = 0.5f0 * (p_ll + p_rr)
    p_div_rho_ln_ratio = Trixi.ln_ratio(p_div_rho_ll, p_div_rho_rr)

    # Calculate fluxes depending on orientation
    if orientation == 1
        f1 = rho_mean * v1_avg
        f2 = f1 * v1_avg + p_avg
        f3 = f1 * v2_avg
        f4 = f1 * v3_avg
        f5 = f1 * (p_div_rho_ln_ratio - gamma - gamma_m1 * ln_rho_avg) +
             gamma_m1 * rho_avg * v1_avg
    else
        f1 = rho_mean * v2_avg
        f2 = f1 * v1_avg
        f3 = f1 * v2_avg + p_avg
        f4 = f1 * v3_avg
        f5 = f1 * (p_div_rho_ln_ratio - gamma - gamma_m1 * ln_rho_avg) +
             gamma_m1 * rho_avg * v2_avg
    end

    return SVector(f1, f2, f3, f4, f5)
end
#=
@inline function flux_euler_energy_con(prim_ll, prim_rr, normal_direction::AbstractVector, i::Integer,
                              equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D)

    # Unpack left and right state
    rho_ll, v1_ll, v2_ll, v3_ll, p_ll = view(prim_ll, (5*i-4):(5*i))
    rho_rr, v1_rr, v2_rr, v3_rr, p_rr = view(prim_rr, (5*i-4):(5*i))
    gamma = equations.gammas[i]
    gamma_m1 = gamma - 1

    p_div_rho_ll = p_ll / rho_ll
    p_div_rho_rr = p_rr / rho_rr
    v_dot_n_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2] +
                 v3_ll * normal_direction[3]
    v_dot_n_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2] +
                 v3_rr * normal_direction[3]

    # Compute the necessary mean values
    rho_mean = ln_mean(rho_ll, rho_rr)
    ln_rho_avg = 0.5f0 * log(rho_ll * rho_rr)
    rho_avg = 0.5f0 * (rho_ll + rho_rr)
    v1_avg = 0.5f0 * (v1_ll + v1_rr)
    v2_avg = 0.5f0 * (v2_ll + v2_rr)
    v3_avg = 0.5f0 * (v3_ll + v3_rr)
    v_dot_n_avg = 0.5f0 * (v_dot_n_ll + v_dot_n_rr)
    p_avg = 0.5f0 * (p_ll + p_rr)
    p_div_rho_ln_ratio = Trixi.ln_ratio(p_div_rho_ll, p_div_rho_rr)

    # Calculate fluxes depending on orientation
    f1 = rho_mean * v_dot_n_avg
    f2 = f1 * v1_avg + p_avg * normal_direction[1]
    f3 = f1 * v2_avg + p_avg * normal_direction[2]
    f4 = f1 * v3_avg + p_avg * normal_direction[3]
    f5 = f1 * (p_div_rho_ln_ratio - gamma - gamma_m1 * ln_rho_avg) +
         gamma_m1 * rho_avg * v_dot_n_avg

    return SVector(f1, f2, f3, f4, f5)
end
=#

# Rotate normal vector to x-axis; normal, tangent1 and tangent2 need to be orthonormal
# Called inside `FluxRotated` in `numerical_fluxes.jl` so the directions
# has been normalized prior to this rotation of the state vector
@inline function rotate_to_x(u, normal_vector, tangent1, tangent2,
                             equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D)
    # Multiply with [ 1   0        0       0   0;
    #                 0   ―  normal_vector ―   0;
    #                 0   ―    tangent1    ―   0;
    #                 0   ―    tangent2    ―   0;
    #                 0   0        0       0   1 ]
    return SVector(u[1],
                   normal_vector[1] * u[2] + normal_vector[2] * u[3] +
                   normal_vector[3] * u[4],
                   tangent1[1] * u[2] + tangent1[2] * u[3] + tangent1[3] * u[4],
                   tangent2[1] * u[2] + tangent2[2] * u[3] + tangent2[3] * u[4],
                   u[5])
end

# Rotate x-axis to normal vector; normal, tangent1 and tangent2 need to be orthonormal
# Called inside `FluxRotated` in `numerical_fluxes.jl` so the directions
# has been normalized prior to this back-rotation of the state vector
@inline function rotate_from_x(u, normal_vector, tangent1, tangent2,
                               equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D)
    # Multiply with [ 1        0          0        0      0;
    #                 0        |          |        |      0;
    #                 0  normal_vector tangent1 tangent2  0;
    #                 0        |          |        |      0;
    #                 0        0          0        0      1 ]
    return SVector(u[1],
                   normal_vector[1] * u[2] + tangent1[1] * u[3] + tangent2[1] * u[4],
                   normal_vector[2] * u[2] + tangent1[2] * u[3] + tangent2[2] * u[4],
                   normal_vector[3] * u[2] + tangent1[3] * u[3] + tangent2[3] * u[4],
                   u[5])
end

@inline function flux_euler_noncon_dissipation_winters_etal(prim_ll, prim_rr,
                                                            normal_direction::AbstractVector, i::Integer,
                                                            equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D)
    gamma = equations.gammas[i]
    inv_gamma_minus_one = equations.inv_gammas_minus_one[i]

    normal_direction_ = SVector(normal_direction[1], normal_direction[2], 0)
    norm_ = norm(normal_direction_)
    normal_vector = normal_direction_ / norm_

    rho_ll, v1_ll, v2_ll, v3_ll, p_ll = view(prim_ll, (5*i-4):(5*i))
    rho_rr, v1_rr, v2_rr, v3_rr, p_rr = view(prim_rr, (5*i-4):(5*i))

    u_ll_ = SVector(rho_ll, rho_ll * v1_ll, rho_ll * v2_ll, rho_ll * v3_ll, p_ll)
    u_rr_ = SVector(rho_rr, rho_rr * v1_rr, rho_rr * v2_rr, rho_rr * v3_rr, p_rr)
    # Step 1:
    # Rotate solution into the appropriate direction

    # Some vector that can't be identical to normal_vector (unless normal_vector == 0)
    tangent1 = SVector(normal_direction_[2], normal_direction_[3], -normal_direction_[1])
    # Orthogonal projection
    tangent1 -= dot(normal_vector, tangent1) * normal_vector
    tangent1 = normalize(tangent1)

    # Third orthogonal vector
    tangent2 = normalize(cross(normal_direction_, tangent1))

    u_ll_rotated = rotate_to_x(u_ll_, normal_vector, tangent1, tangent2, equations)
    u_rr_rotated = rotate_to_x(u_rr_, normal_vector, tangent1, tangent2, equations)

    # Step 2:
    # Compute the averages using the rotated variables
    v1_ll = u_ll_rotated[2] / u_ll_rotated[1]
    v2_ll = u_ll_rotated[3] / u_ll_rotated[1]
    v3_ll = u_ll_rotated[4] / u_ll_rotated[1]
    v1_rr = u_rr_rotated[2] / u_rr_rotated[1]
    v2_rr = u_rr_rotated[3] / u_rr_rotated[1]
    v3_rr = u_rr_rotated[4] / u_rr_rotated[1]

    rho_e_total_ll = 0.5f0 * rho_ll * (v1_ll^2 + v2_ll^2 + v3_ll^2) + p_ll * inv_gamma_minus_one
    rho_e_total_rr = 0.5f0 * rho_rr * (v1_rr^2 + v2_rr^2 + v3_rr^2) + p_rr * inv_gamma_minus_one

    b_ll = rho_ll / (2 * p_ll)
    b_rr = rho_rr / (2 * p_rr)

    rho_log = ln_mean(rho_ll, rho_rr)
    b_log = ln_mean(b_ll, b_rr)
    v1_avg = 0.5f0 * (v1_ll + v1_rr)
    v2_avg = 0.5f0 * (v2_ll + v2_rr)
    v3_avg = 0.5f0 * (v3_ll + v3_rr)
    p_avg = 0.5f0 * (rho_ll + rho_rr) / (b_ll + b_rr)
    v_squared_bar = v1_ll * v1_rr + v2_ll * v2_rr + v3_ll * v3_rr
    h_bar = gamma / (2 * b_log * (gamma - 1)) + 0.5f0 * v_squared_bar
    c_bar = sqrt(gamma * p_avg / rho_log)

    # Step 3:
    # Build the dissipation term as given in Appendix A of the paper 
    # - A. R. Winters, D. Derigs, G. Gassner, S. Walch, A uniquely defined entropy stable matrix dissipation operator 
    # for high Mach number ideal MHD and compressible Euler simulations (2017). Journal of Computational Physics.
    # [DOI: 10.1016/j.jcp.2016.12.006](https://doi.org/10.1016/j.jcp.2016.12.006).

    # Get entropy variables jump in the rotated variables
    entropy_classic_ll = cons2entropy_euler_classic(SVector(rho_ll, rho_ll * v1_ll, rho_ll * v2_ll, rho_ll * v3_ll, rho_e_total_ll), equations)
    entropy_classic_rr = cons2entropy_euler_classic(SVector(rho_rr, rho_rr * v1_rr, rho_rr * v2_rr, rho_rr * v3_rr, rho_e_total_rr), equations)
    w_jump = entropy_classic_rr - entropy_classic_ll

    # Entries of the diagonal scaling matrix where D = ABS(\Lambda)T
    lambda_1 = abs(v1_avg - c_bar) * rho_log / (2 * gamma)
    lambda_2 = abs(v1_avg) * rho_log * (gamma - 1) / gamma
    lambda_3 = abs(v1_avg) * p_avg # scaled repeated eigenvalue in the tangential direction
    lambda_5 = abs(v1_avg + c_bar) * rho_log / (2 * gamma)
    D = SVector(lambda_1, lambda_2, lambda_3, lambda_3, lambda_5)

    # Entries of the right eigenvector matrix (others have already been precomputed)
    r21 = v1_avg - c_bar
    r25 = v1_avg + c_bar
    r51 = h_bar - v1_avg * c_bar
    r52 = 0.5f0 * v_squared_bar
    r55 = h_bar + v1_avg * c_bar

    # Build R and transpose of R matrices
    R = @SMatrix [[1;; 1;; 0;; 0;; 1];
                  [r21;; v1_avg;; 0;; 0;; r25];
                  [v2_avg;; v2_avg;; 1;; 0;; v2_avg];
                  [v3_avg;; v3_avg;; 0;; 1;; v3_avg];
                  [r51;; r52;; v2_avg;; v3_avg;; r55]]

    RT = @SMatrix [[1;; r21;; v2_avg;; v3_avg;; r51];
                   [1;; v1_avg;; v2_avg;; v3_avg;; r52];
                   [0;; 0;; 1;; 0;; v2_avg];
                   [0;; 0;; 0;; 1;; v3_avg];
                   [1;; r25;; v2_avg;; v3_avg;; r55]]

    # Compute the dissipation term R * D * R^T * [[w]] from right-to-left

    # First comes R^T * [[w]]
    diss = RT * w_jump
    # Next multiply with the eigenvalues and Barth scaling
    diss = D .* diss
    # Finally apply the remaining eigenvector matrix
    diss = R * diss

    original_dissipation = -0.5f0 * rotate_from_x(diss, normal_vector, tangent1, tangent2, equations) * norm_
    
    u_ll___ = prim2cons(prim_ll, equations)
    #ent_ll_ = cons2entropy(u_ll___, equations)
    entropy_classic_ll_2 = cons2entropy_euler_classic(view(u_ll___, (5*i-4):(5*i)), equations)

    #u_rr___ = prim2cons(prim_rr, equations)
    #ent_rr_ = cons2entropy(u_rr___, equations)
    #entropy_classic_rr_2 = cons2entropy_euler_classic(view(u_rr___, (5*i-4):(5*i)), equations)

    dissipation_entropy = -dot(entropy_classic_ll_2, original_dissipation)
    #back_converted = dot(SVector(original_dissipation[1], original_dissipation[2], original_dissipation[3], original_dissipation[4], dissipation_entropy), view(ent_ll_, (5*i-4):(5*i)))
    #dissipation_entropy_rr = -dot(entropy_classic_rr_2, original_dissipation)
    #back_converted_rr = dot(SVector(original_dissipation[1], original_dissipation[2], original_dissipation[3], original_dissipation[4], dissipation_entropy_rr), view(ent_rr_, (5*i-4):(5*i)))
    #=
    println(original_dissipation[5])
    println(back_converted)
    println(back_converted_rr)
    println()
    =#
    return SVector(original_dissipation[1], original_dissipation[2], 
                   original_dissipation[3], original_dissipation[4], dissipation_entropy)

    #=
    rho_log = ln_mean(rho_ll, rho_rr)
    b_log = ln_mean(b_ll, b_rr)
    v1_avg = 0.5f0 * (v1_ll + v1_rr)
    v2_avg = 0.5f0 * (v2_ll + v2_rr)
    v3_avg = 0.5f0 * (v3_ll + v3_rr)
    p_avg = 0.5f0 * (rho_ll + rho_rr) / (b_ll + b_rr) # 2 * b_avg = b_ll + b_rr
    v_squared_bar = v1_ll * v1_rr + v2_ll * v2_rr + v3_ll * v3_rr
    h_bar = gamma / (2 * b_log * (gamma - 1)) + 0.5f0 * v_squared_bar
    c_bar = sqrt(gamma * p_avg / rho_log)

    v_avg_normal = dot(SVector(v1_avg, v2_avg), unit_normal_direction)

    lambda_1 = abs(v_avg_normal - c_bar) * rho_log / (2 * gamma)
    lambda_2 = abs(v_avg_normal) * rho_log * (gamma - 1) / gamma
    lambda_3 = abs(v_avg_normal + c_bar) * rho_log / (2 * gamma)
    lambda_4 = abs(v_avg_normal) * p_avg

    v1_minus_c = v1_avg - c_bar * unit_normal_direction[1]
    v2_minus_c = v2_avg - c_bar * unit_normal_direction[2]
    v3_minus_c = v3_avg
    v1_plus_c = v1_avg + c_bar * unit_normal_direction[1]
    v2_plus_c = v2_avg + c_bar * unit_normal_direction[2]
    v3_plus_c = v3_avg
    v1_tangential = v1_avg - v_avg_normal * unit_normal_direction[1]
    v2_tangential = v2_avg - v_avg_normal * unit_normal_direction[2]
    v3_tangential = v3_avg


    entropy_classic_ll = cons2entropy_euler_classic(SVector(rho_ll, rho_ll * v1_ll, rho_ll * v2_ll, rho_ll * v3_ll, rho_e_total_ll), i, equations)
    entropy_classic_rr = cons2entropy_euler_classic(SVector(rho_rr, rho_rr * v1_rr, rho_rr * v2_rr, rho_rr * v3_rr, rho_e_total_rr), i, equations)

    entropy_vars_jump = entropy_classic_ll - entropy_classic_rr

    entropy_var_rho_jump, entropy_var_rho_v1_jump,
    entropy_var_rho_v2_jump, entropy_var_rho_v3_jump, entropy_var_rho_e_jump = entropy_vars_jump

    velocity_minus_c_dot_entropy_vars_jump = v1_minus_c * entropy_var_rho_v1_jump +
                                             v2_minus_c * entropy_var_rho_v2_jump +
                                             v3_minus_c * entropy_var_rho_v3_jump
    velocity_plus_c_dot_entropy_vars_jump = v1_plus_c * entropy_var_rho_v1_jump +
                                            v2_plus_c * entropy_var_rho_v2_jump +
                                            v3_plus_c * entropy_var_rho_v3_jump
    velocity_avg_dot_vjump = v1_avg * entropy_var_rho_v1_jump +
                             v2_avg * entropy_var_rho_v2_jump +
                             v3_avg * entropy_var_rho_v3_jump

    w1 = lambda_1 * (entropy_var_rho_jump + velocity_minus_c_dot_entropy_vars_jump +
          (h_bar - c_bar * v_avg_normal) * entropy_var_rho_e_jump)
    w2 = lambda_2 * (entropy_var_rho_jump + velocity_avg_dot_vjump +
          v_squared_bar / 2 * entropy_var_rho_e_jump)
    w3 = lambda_3 * (entropy_var_rho_jump + velocity_plus_c_dot_entropy_vars_jump +
          (h_bar + c_bar * v_avg_normal) * entropy_var_rho_e_jump)

    entropy_var_v_normal_jump = dot(SVector(entropy_var_rho_v1_jump,
                                        entropy_var_rho_v2_jump),
                                    unit_normal_direction)


    dissipation_rho = w1 + w2 + w3

    dissipation_rho_v1 = (w1 * v1_minus_c +
                        w2 * v1_avg +
                        w3 * v1_plus_c +
                        lambda_4 * (entropy_var_rho_v1_jump -
                        unit_normal_direction[1] * entropy_var_v_normal_jump +
                        entropy_var_rho_e_jump * v1_tangential))

    dissipation_rho_v2 = (w1 * v2_minus_c +
                        w2 * v2_avg +
                        w3 * v2_plus_c +
                        lambda_4 * (entropy_var_rho_v2_jump -
                        unit_normal_direction[2] * entropy_var_v_normal_jump +
                        entropy_var_rho_e_jump * v2_tangential))

    dissipation_rho_v3 = (w1 * v3_minus_c +
                        w2 * v3_avg +
                        w3 * v3_plus_c +
                        lambda_4 * (entropy_var_rho_v3_jump +
                        entropy_var_rho_e_jump * v3_tangential))

    v_tangential_dot_entropy_vars_jump = v1_tangential * entropy_var_rho_v1_jump +
                                         v2_tangential * entropy_var_rho_v2_jump +
                                         v3_tangential * entropy_var_rho_v3_jump

    dissipation_rhoe = (w1 * (h_bar - c_bar * v_avg_normal) +
                        w2 * 0.5f0 * v_squared_bar +
                        w3 * (h_bar + c_bar * v_avg_normal) +
                        lambda_4 * (v_tangential_dot_entropy_vars_jump +
                         entropy_var_rho_e_jump *
                         (v1_avg^2 + v2_avg^2 + v3_avg^2 - v_avg_normal^2)))
        

    original_dissipation = -0.5f0 *
           SVector(dissipation_rho, dissipation_rho_v1, dissipation_rho_v2, dissipation_rho_v3,
                   dissipation_rhoe) * norm_
    dissipation_entropy = -dot(entropy_classic_ll, original_dissipation)
    #dissipation_rho = 0.0
    #dissipation_rho_v1 = 0.0
    #dissipation_rho_v2 = 0.0
    #dissipation_rho_v3 = 0.0
    #dissipation_entropy = 0.0
    return 2.0*SVector(dissipation_rho, dissipation_rho_v1, dissipation_rho_v2, dissipation_rho_v3, dissipation_entropy)

=#
#=
norm_ = norm(normal_direction)
    unit_normal_direction = normal_direction / norm_

    rho_ll, v1_ll, v2_ll, p_ll = cons2prim(u_ll, equations)
    rho_rr, v1_rr, v2_rr, p_rr = cons2prim(u_rr, equations)

    b_ll = rho_ll / (2 * p_ll)
    b_rr = rho_rr / (2 * p_rr)

    rho_log = ln_mean(rho_ll, rho_rr)
    b_log = ln_mean(b_ll, b_rr)
    v1_avg = 0.5f0 * (v1_ll + v1_rr)
    v2_avg = 0.5f0 * (v2_ll + v2_rr)
    p_avg = 0.5f0 * (rho_ll + rho_rr) / (b_ll + b_rr) # 2 * b_avg = b_ll + b_rr
    v_squared_bar = v1_ll * v1_rr + v2_ll * v2_rr
    h_bar = gamma / (2 * b_log * (gamma - 1)) + 0.5f0 * v_squared_bar
    c_bar = sqrt(gamma * p_avg / rho_log)

    v_avg_normal = dot(SVector(v1_avg, v2_avg), unit_normal_direction)

    lambda_1 = abs(v_avg_normal - c_bar) * rho_log / (2 * gamma)
    lambda_2 = abs(v_avg_normal) * rho_log * (gamma - 1) / gamma
    lambda_3 = abs(v_avg_normal + c_bar) * rho_log / (2 * gamma)
    lambda_4 = abs(v_avg_normal) * p_avg

    v1_minus_c = v1_avg - c_bar * unit_normal_direction[1]
    v2_minus_c = v2_avg - c_bar * unit_normal_direction[2]
    v1_plus_c = v1_avg + c_bar * unit_normal_direction[1]
    v2_plus_c = v2_avg + c_bar * unit_normal_direction[2]
    v1_tangential = v1_avg - v_avg_normal * unit_normal_direction[1]
    v2_tangential = v2_avg - v_avg_normal * unit_normal_direction[2]

    entropy_vars_jump = cons2entropy(u_rr, equations) - cons2entropy(u_ll, equations)
    entropy_var_rho_jump, entropy_var_rho_v1_jump,
    entropy_var_rho_v2_jump, entropy_var_rho_e_jump = entropy_vars_jump

    velocity_minus_c_dot_entropy_vars_jump = v1_minus_c * entropy_var_rho_v1_jump +
                                             v2_minus_c * entropy_var_rho_v2_jump
    velocity_plus_c_dot_entropy_vars_jump = v1_plus_c * entropy_var_rho_v1_jump +
                                            v2_plus_c * entropy_var_rho_v2_jump
    velocity_avg_dot_vjump = v1_avg * entropy_var_rho_v1_jump +
                             v2_avg * entropy_var_rho_v2_jump
    w1 = lambda_1 * (entropy_var_rho_jump + velocity_minus_c_dot_entropy_vars_jump +
          (h_bar - c_bar * v_avg_normal) * entropy_var_rho_e_jump)
    w2 = lambda_2 * (entropy_var_rho_jump + velocity_avg_dot_vjump +
          v_squared_bar / 2 * entropy_var_rho_e_jump)
    w3 = lambda_3 * (entropy_var_rho_jump + velocity_plus_c_dot_entropy_vars_jump +
          (h_bar + c_bar * v_avg_normal) * entropy_var_rho_e_jump)

    entropy_var_v_normal_jump = dot(SVector(entropy_var_rho_v1_jump,
                                            entropy_var_rho_v2_jump),
                                    unit_normal_direction)

    dissipation_rho = w1 + w2 + w3

    dissipation_rho_v1 = (w1 * v1_minus_c +
                          w2 * v1_avg +
                          w3 * v1_plus_c +
                          lambda_4 * (entropy_var_rho_v1_jump -
                           unit_normal_direction[1] * entropy_var_v_normal_jump +
                           entropy_var_rho_e_jump * v1_tangential))

    dissipation_rho_v2 = (w1 * v2_minus_c +
                          w2 * v2_avg +
                          w3 * v2_plus_c +
                          lambda_4 * (entropy_var_rho_v2_jump -
                           unit_normal_direction[2] * entropy_var_v_normal_jump +
                           entropy_var_rho_e_jump * v2_tangential))

    v_tangential_dot_entropy_vars_jump = v1_tangential * entropy_var_rho_v1_jump +
                                         v2_tangential * entropy_var_rho_v2_jump

    dissipation_rhoe = (w1 * (h_bar - c_bar * v_avg_normal) +
                        w2 * 0.5f0 * v_squared_bar +
                        w3 * (h_bar + c_bar * v_avg_normal) +
                        lambda_4 * (v_tangential_dot_entropy_vars_jump +
                         entropy_var_rho_e_jump *
                         (v1_avg^2 + v2_avg^2 - v_avg_normal^2)))

    return -0.5f0 *
           SVector(dissipation_rho, dissipation_rho_v1, dissipation_rho_v2,
                   dissipation_rhoe) * norm_
=#

end

@inline flux_noncon_empty(u_ll, u_rr, orientation_or_normal_direction, equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D) = zeros(SVector{nvariables(equations), typeof(equations.gammas[1])})

@inline function flux_glm_upwind(
    u_ll,
    u_rr,
    orientation::Integer,
    equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D,
)
    c = equations.speed_of_light
    c_e = equations.c_e
    c_b = equations.c_b

    n = 5*ncomponents(equations)
    u_sum = view(u_ll, (n+1):(n+8)) + view(u_rr, (n+1):(n+8))
    u_diff = view(u_ll, (n+1):(n+8)) - view(u_rr, (n+1):(n+8))

    if orientation == 1
        f1 = 0.5f0 * c * c_e * (u_diff[1] + c * u_sum[7])
        f2 = 0.5f0 * c * (u_diff[2] + c * u_sum[6])
        f3 = 0.5f0 * c * (u_diff[3] - c * u_sum[5])
        f4 = 0.5f0 * c_b * (u_sum[8] + c * u_diff[4])
        f5 = 0.5f0 * (-u_sum[3] + c * u_diff[5])
        f6 = 0.5f0 * (u_sum[2] + c * u_diff[6])
        f7 = 0.5f0 * c_e * (u_sum[1] + c * u_diff[7])
        f8 = 0.5f0 * c_b * c * (u_diff[8] + c * u_sum[4])
    else
        f1 = 0.5f0 * c * (u_diff[1] - c * u_sum[6])
        f2 = 0.5f0 * c * c_e * (u_diff[2] + c * u_sum[7])
        f3 = 0.5f0 * c * (u_diff[3] + c * u_sum[4])
        f4 = 0.5f0 * (u_sum[3] + c * u_diff[4])
        f5 = 0.5f0 * c_b * (u_sum[8] + c * u_diff[5])
        f6 = 0.5f0 * (-u_sum[1] + c * u_diff[6])
        f7 = 0.5f0 * c_e * (u_sum[2] + c * u_diff[7])
        f8 = 0.5f0 * c * c_b * (u_diff[8] + c * u_sum[5])
    end

    return SVector(f1, f2, f3, f4, f5, f6, f7, f8)
end
#=
@inline function flux_glm_upwind(
    u_ll,
    u_rr,
    normal_direction::AbstractVector,
    equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D,
)
    c = equations.speed_of_light
    c_e = equations.c_e
    c_b = equations.c_b

    n = 5*ncomponents(equations)
    u_sum = view(u_ll, (n+1):(n+8)) + view(u_rr, (n+1):(n+8))
    u_diff = view(u_ll, (n+1):(n+8)) - view(u_rr, (n+1):(n+8))

    f1_1 = 0.5f0 * c * c_e * (normal_direction[1] * u_diff[1]  + c * u_sum[7])
    f2_1 = 0.5f0 * c * (normal_direction[1] * u_diff[2] + c * u_sum[6])
    f3_1 = 0.5f0 * c * (normal_direction[1] * u_diff[3] - c * u_sum[5])
    f4_1 = 0.5f0 * c_b * (u_sum[8] + c * normal_direction[1] * u_diff[4])
    f5_1 = 0.5f0 * (-u_sum[3] + c * normal_direction[1] * u_diff[5])
    f6_1 = 0.5f0 * (u_sum[2] + c * normal_direction[1] * u_diff[6])
    f7_1 = 0.5f0 * c_e * (u_sum[1] + c * normal_direction[1] * u_diff[7])
    f8_1 = 0.5f0 * c_b * c * (normal_direction[1] * u_diff[8] + c * u_sum[4])

    f1_2 = 0.5f0 * c * (normal_direction[2] * u_diff[1] - c * u_sum[6])
    f2_2 = 0.5f0 * c * c_e * (normal_direction[2] * u_diff[2] + c * u_sum[7])
    f3_2 = 0.5f0 * c * (normal_direction[2] * u_diff[3] + c * u_sum[4])
    f4_2 = 0.5f0 * (u_sum[3] + c * normal_direction[2] * u_diff[4])
    f5_2 = 0.5f0 * c_b * (u_sum[8] + c * normal_direction[2] * u_diff[5])
    f6_2 = 0.5f0 * (-u_sum[1] + c * normal_direction[2] * u_diff[6])
    f7_2 = 0.5f0 * c_e * (u_sum[2] + c * normal_direction[2] * u_diff[7])
    f8_2 = 0.5f0 * c * c_b * (normal_direction[2] * u_diff[8] + c * u_sum[5])

    f1_3 = 0.5f0 * c * (normal_direction[3] * u_diff[1] + c * u_sum[5])
    f2_3 = 0.5f0 * c * (normal_direction[3] * u_diff[2] - c * u_sum[4])
    f3_3 = 0.5f0 * c * c_e * (normal_direction[3] * u_diff[3] + c * u_sum[7])
    f4_3 = 0.5f0 * (-u_sum[2] + c * normal_direction[3] * u_diff[4])
    f5_3 = 0.5f0 * (u_sum[1] + c * normal_direction[3] * u_diff[5])
    f6_3 = 0.5f0 * c_b * (u_sum[8] + c * normal_direction[3] * u_diff[6])
    f7_3 = 0.5f0 * c_e * (u_sum[3] + c * normal_direction[3] * u_diff[7])
    f8_3 = 0.5f0 * c * c_b * (normal_direction[3] * u_diff[8] + c * u_sum[6])

    f1 = f1_1 * normal_direction[1] + f1_2 * normal_direction[2] + f1_3 * normal_direction[3]
    f2 = f2_1 * normal_direction[1] + f2_2 * normal_direction[2] + f2_3 * normal_direction[3]
    f3 = f3_1 * normal_direction[1] + f3_2 * normal_direction[2] + f3_3 * normal_direction[3]
    f4 = f4_1 * normal_direction[1] + f4_2 * normal_direction[2] + f4_3 * normal_direction[3]
    f5 = f5_1 * normal_direction[1] + f5_2 * normal_direction[2] + f5_3 * normal_direction[3]
    f6 = f6_1 * normal_direction[1] + f6_2 * normal_direction[2] + f6_3 * normal_direction[3]
    f7 = f7_1 * normal_direction[1] + f7_2 * normal_direction[2] + f7_3 * normal_direction[3]
    f8 = f8_1 * normal_direction[1] + f8_2 * normal_direction[2] + f8_3 * normal_direction[3]

    return SVector(f1, f2, f3, f4, f5, f6, f7, f8)
end
=#
@inline function flux_glm_central(u_ll, u_rr, orientation_or_normal_direction,
                              equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D)
    return 0.5f0 * (flux_glm_maxwell(u_ll, orientation_or_normal_direction, equations) + flux_glm_maxwell(u_rr, orientation_or_normal_direction, equations))
end

min_max_speed_naive(u_ll, u_rr, orientation::Integer, equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D) =
    max(1, equations.c_e, equations.c_b) * (-equations.speed_of_light, equations.speed_of_light)

min_max_speed_naive(u_ll, u_rr, normal_direction::AbstractVector, equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D) =
    max(1, equations.c_e, equations.c_b) * norm(normal_direction) * (-equations.speed_of_light, equations.speed_of_light)

max_abs_speeds(u, equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D) =
    (max(1, equations.c_e, equations.c_b) * equations.speed_of_light, max(1, equations.c_e, equations.c_b) * equations.speed_of_light,
     max(1, equations.c_e, equations.c_b) * equations.speed_of_light)

max_abs_speed_naive(u_ll, u_rr, orientation::Integer, equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D) =
    max(1, equations.c_e, equations.c_b) * equations.speed_of_light

max_abs_speed(u_ll, u_rr, orientation::Integer, equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D) =
    max(1, equations.c_e, equations.c_b) * equations.speed_of_light

max_abs_speed_naive(u_ll, u_rr, normal_direction::AbstractVector, equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D) =
    max(1, equations.c_e, equations.c_b) * norm(normal_direction) * equations.speed_of_light


function source_term_lorentz(u, x, t, equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D)
    prim = cons2prim(u, equations)
    sources_euler = ntuple(i -> source_term_lorentz_euler(prim, x, t, i, equations), ncomponents(equations))
    sources_glm = source_term_lorentz_glm(u, x, t, equations)
    return vcat(sources_euler..., sources_glm)
end

function source_term_lorentz_corrected(u, x, t, equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D)
    prim = cons2prim(u, equations)
    sources_euler = ntuple(i -> source_term_lorentz_corrected_euler(prim, x, t, i, equations), ncomponents(equations))
    sources_glm = source_term_lorentz_glm(u, x, t, equations)
    return vcat(sources_euler..., sources_glm)
end

function source_term_lorentz_corrected_2(u, x, t, equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D)
    prim = cons2prim(u, equations)
    sources_euler = ntuple(i -> source_term_lorentz_corrected_euler_2(prim, x, t, i, equations), ncomponents(equations))
    sources_glm = source_term_lorentz_glm(u, x, t, equations)
    return vcat(sources_euler..., sources_glm)
end

function source_term_lorentz_corrected_relaxation_2(u, x, t, equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D)
    prim = cons2prim(u, equations)
    sources_euler = ntuple(i -> source_term_lorentz_corrected_euler_2(prim, x, t, i, equations), ncomponents(equations))
    sources_glm = source_term_lorentz_relaxation_glm(u, x, t, equations)
    return vcat(sources_euler..., sources_glm)
end

function source_term_lorentz_euler(prim, x, t, i, equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D)
    charge_mass_ratio = equations.charge_mass_ratios[i]
    gas_constant = equations.gas_constants[i]
    rho, v1, v2, v3, p = view(prim, (5*i-4):(5*i))
    E1, E2, E3, B1, B2, B3 = prim[end-7], prim[end-6], prim[end-5], prim[end-4], prim[end-3], prim[end-2]

    s1 = 0
    s2 = charge_mass_ratio * rho * (E1 + v2 * B3 - v3 * B2)
    s3 = charge_mass_ratio * rho * (E2 + v3 * B1 - v1 * B3)
    s4 = charge_mass_ratio * rho * (E3 + v1 * B2 - v2 * B1)
    s5 = 0
    return SVector(s1, s2, s3, s4, s5)
end

function source_term_lorentz_corrected_euler(prim, x, t, i, equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D)
    charge_mass_ratio = equations.charge_mass_ratios[i]
    rho, v1, v2, v3, p = view(prim, (5*i-4):(5*i))
    E1, E2, E3, B1, B2, B3, psi_E = prim[end-7], prim[end-6], prim[end-5], prim[end-4], prim[end-3], prim[end-2], prim[end-1]
    gamma_m1 = equations.gammas[i] - 1

    s1 = 0
    s2 = charge_mass_ratio * rho * (E1 + v2 * B3 - v3 * B2)
    s3 = charge_mass_ratio * rho * (E2 + v3 * B1 - v1 * B3)
    s4 = charge_mass_ratio * rho * (E3 + v1 * B2 - v2 * B1)
    s5 = -gamma_m1 * equations.c_sqr * equations.c_e * charge_mass_ratio * rho^2 * psi_E / p
    return SVector(s1, s2, s3, s4, s5)
end

function source_term_lorentz_corrected_euler_2(prim, x, t, i, equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D)
    total_density = sum(densities(prim, equations))
    total_charge_density = charge_density(prim, equations)
    charge = total_charge_density / total_density
    charge_mass_ratio = equations.charge_mass_ratios[i]
    rho, v1, v2, v3, p = view(prim, (5*i-4):(5*i))
    E1, E2, E3, B1, B2, B3, psi_E = prim[end-7], prim[end-6], prim[end-5], prim[end-4], prim[end-3], prim[end-2], prim[end-1]
    gamma_m1 = equations.gammas[i] - 1

    s1 = 0
    s2 = charge_mass_ratio * rho * (E1 + v2 * B3 - v3 * B2)
    s3 = charge_mass_ratio * rho * (E2 + v3 * B1 - v1 * B3)
    s4 = charge_mass_ratio * rho * (E3 + v1 * B2 - v2 * B1)
    s5 = -gamma_m1 * equations.c_sqr * equations.c_e * charge * rho^2 * psi_E / p
    return SVector(s1, s2, s3, s4, s5)
end


function source_term_lorentz_glm(u, x, t, equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D)
    current_density = Trixi.current_density(u, equations)
    charge_density = Trixi.charge_density(u, equations)
    inv_permittivity = inv(equations.permittivity)

    s1 = -current_density[1] * inv_permittivity
    s2 = -current_density[2] * inv_permittivity
    s3 = -current_density[3] * inv_permittivity
    s4 = 0
    s5 = 0
    s6 = 0
    s7 = equations.c_e * inv_permittivity * charge_density
    s8 = 0
    return SVector(s1, s2, s3, s4, s5, s6, s7, s8)
end

function source_term_lorentz_relaxation_glm(u, x, t, equations::GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D)
    current_density = Trixi.current_density(u, equations)
    charge_density = Trixi.charge_density(u, equations)
    inv_permittivity = inv(equations.permittivity)

    s1 = -current_density[1] * inv_permittivity
    s2 = -current_density[2] * inv_permittivity
    s3 = -current_density[3] * inv_permittivity
    s4 = 0
    s5 = 0
    s6 = 0
    s7 = equations.c_e * inv_permittivity * charge_density - 0.1 * equations.c_e * u[end-1]
    s8 = - equations.c_b * 0.1 * u[end]

    return SVector(s1, s2, s3, s4, s5, s6, s7, s8)
end

@inline function density_pressure(u, equations::Trixi.GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D)
    rhos = densities(u, equations)
    entr_s = entropies(u, equations)
    rho_times_p = rhos.^(equations.gammas .+ 1) .* exp.(entr_s ./ rhos)
    return minimum(rho_times_p)
end

@inline function density_pressure_alt(u, equations::Trixi.GlmMultiFluid5MomentPlasmaEquationsEntropyPseudo3D)
    rhos = densities(u, equations)
    entr_s = entropies(u, equations)
    E1, E2, E3 = electric_field(u, equations)
    B1, B2, B3 = magnetic_field(u, equations)
    chi_E = u[end-1]
    chi_B = u[end]
    EM_pressure = 0.5f0 * ( equations.permittivity * (E1^2 + E2^2 + E3^2 + chi_B^2) + (B1^2 + B2^2 + B3^2 + chi_E^2) / equations.permeability)
    rho_times_p = rhos.^(equations.gammas .+ 1) .* exp.(entr_s ./ rhos)
    return sum(rhos)
end

end # @muladd
