# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

struct GLMMultiFluid5MomentPlasmaEquations{NVARS, NCOMP, RealT <: Real} <: AbstractGLMMultiFluid5MomentPlasmaEquations{2, NVARS, NCOMP}
    gammas::SVector{NCOMP, RealT}               # specific heat for each species
    inv_gammas_minus_one::SVector{NCOMP, RealT}  # = inv(gamma - 1); can be used to write slow divisions as fast multiplications
    charge_per_mass::SVector{NCOMP, RealT}      # charge of one particle of each species divided by its mass
    speed_of_light::RealT                       # c
    c_h::RealT                                  # GLM cleaning speed
    function GLMMultiFluid5MomentPlasmaEquations(c = 299_792_458.0, c_h = 1.0)
        new{typeof(c_h)}(c, c_h)
    end
    function GLMMultiFluid5MomentPlasmaEquations{NVARS, NCOMP, RealT}(gammas::SVector{NCOMP, RealT},
                                                                    charge_per_mass::SVector{NCOMP, RealT},
                                                                    c::RealT, c_h::RealT) where {
                                                                                                                   NVARS,
                                                                                                                   NCOMP,
                                                                                                                   RealT <:
                                                                                                                   Real
                                                                                                                   }
        NCOMP >= 1 ||
            throw(DimensionMismatch("`gammas` and `charge_per_mass` have to be filled with at least one value"))

        gammas, inv_gammas_minus_one = promote(gammas, inv.(gammas - 1))

        new(gammas, inv_gammas_minus_one, charge_per_mass, c, c_h)
    end
end

# Convert conservative vaiables to primitive
@inline cons2prim(u, equations::GLMMultiFluid5MomentPlasmaEquations) = u

# Convert conservative variables to entropy variables
@inline cons2entropy(u, equations::GLMMultiFluid5MomentPlasmaEquations) = u

varnames(::typeof(cons2cons), ::GLMMultiFluid5MomentPlasmaEquations) = ("E1", "E2", "B", "Psi")
varnames(::typeof(cons2prim), ::GLMMultiFluid5MomentPlasmaEquations) = ("E1", "E2", "B", "Psi")

function default_analysis_integrals(::GLMMultiFluid5MomentPlasmaEquations)
    (Val(:l2_dive), Val(:l2_e_normal_jump))
end

@inline electric_field(u, equations::GLMMultiFluid5MomentPlasmaEquations) = SVector(u[end-3], u[end-2])

@inline scaled_charge_density(u, x, t, source_terms::Nothing, equations::GLMMultiFluid5MomentPlasmaEquations) = 0.0

@inline scaled_charge_density(u, x, t, source_terms, equations::GLMMultiFluid5MomentPlasmaEquations) = source_terms(u, x, t, equations)[end] / equations.c_h^2


@inline function flux(u, orientation::Integer, equations::GLMMultiFluid5MomentPlasmaEquations)
    c_sqr = equations.speed_of_light^2

    if orientation == 1
        f1 = c_sqr * u[4]
        f2 = c_sqr * u[3]
        f3 = u[2]
        f4 = equations.c_h^2 * u[1]
    else
        f1 = -c_sqr * u[3]
        f2 = c_sqr * u[4]
        f3 = -u[1]
        f4 = equations.c_h^2 * u[2]
    end

    return SVector(f1, f2, f3, f4)
end

# Calculates the Euler flux for a single species at a single point
@inline function flux_euler(u, orientation::Integer, equations::CompressibleEulerEquations2D)
    rho, rho_v1, rho_v2, rho_e = u
    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    p = (equations.gamma - 1) * (rho_e - 0.5f0 * (rho_v1 * v1 + rho_v2 * v2))
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
@inline function flux_glm_maxwell(u, orientation::Integer, equations::GLMMultiFluid5MomentPlasmaEquations)
    c_sqr = equations.speed_of_light^2

    if orientation == 1
        f1 = c_sqr * u[4]
        f2 = c_sqr * u[3]
        f3 = u[2]
        f4 = equations.c_h^2 * u[1]
    else
        f1 = -c_sqr * u[3]
        f2 = c_sqr * u[4]
        f3 = -u[1]
        f4 = equations.c_h^2 * u[2]
    end

    return SVector(f1, f2, f3, f4)
end


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

end # @muladd
