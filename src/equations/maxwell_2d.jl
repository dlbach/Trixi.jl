# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent
struct MaxwellEquations2D{RealT <: Real} <: AbstractMaxwellEquations{2, 3}
    speed_of_light::RealT # c

    function MaxwellEquations2D(c::Real = 299_792_458.0)
        new{typeof(c)}(c)
    end
end

varnames(::typeof(cons2cons), ::MaxwellEquations2D) = ("E1", "E2", "B")
varnames(::typeof(cons2prim), ::MaxwellEquations2D) = ("E1", "E2", "B")

@inline electric_field(u, equations::MaxwellEquations2D) = SVector(u[1], u[2])

# Since we do not use the charge density anywhere, we only calculate homogeneous divergence errors
@inline scaled_charge_density(x, u, t, source_terms, equations::MaxwellEquations2D) = 0.0

# Convert conservative variables to primitive
@inline cons2prim(u, equations::MaxwellEquations2D) = u

# Convert conservative variables to entropy variables
@inline cons2entropy(u, equations::MaxwellEquations2D) = u

function default_analysis_integrals(::MaxwellEquations2D)
    (Val(:l2_dive), )
end

@inline function flux(u, orientation::Integer, equations::MaxwellEquations2D)
    c_sqr = equations.speed_of_light^2
    if orientation == 1
        f1 = 0.0f0
        f2 = c_sqr * u[3]
        f3 = u[2]
    else
        f1 = -c_sqr * u[3]
        f2 = 0.0
        f3 = -u[1]
    end

    return Trixi.SVector(f1, f2, f3)
end

@inline function flux_upwind(u_ll,
                             u_rr,
                             orientation::Integer,
                             equations::MaxwellEquations2D)
    c = equations.speed_of_light
    u_sum = u_ll + u_rr
    u_diff = u_ll - u_rr
    if orientation == 1
        f1 = 0.0f0
        f2 = 0.5f0 * c * (u_diff[2] + c * u_sum[3])
        f3 = 0.5f0 * (u_sum[2] + c * u_diff[3])
    else
        f1 = 0.5f0 * c * (u_diff[1] - c * u_sum[3])
        f2 = 0.0f0
        f3 = 0.5f0 * (c * u_diff[3] - u_sum[1])
    end

    return Trixi.SVector(f1, f2, f3)
end

@inline function flux_upwind(u_ll,
                             u_rr,
                             normal_direction::AbstractVector,
                             equations::MaxwellEquations2D)
    c = equations.speed_of_light
    u_sum = u_ll + u_rr
    u_diff = u_ll - u_rr
    f1 = -0.5f0 *
         c *
         normal_direction[2] *
         (-normal_direction[2] * u_diff[1] +
          normal_direction[1] * u_diff[2] +
          c * u_sum[3])
    f2 = 0.5f0 *
         c *
         normal_direction[1] *
         (-normal_direction[2] * u_diff[1] +
          normal_direction[1] * u_diff[2] +
          c * u_sum[3])
    f3 = 0.5f0 * (-normal_direction[2] * u_sum[1] +
          normal_direction[1] * u_sum[2] +
          c * u_diff[3])

    return Trixi.SVector(f1, f2, f3)
end

function boundary_condition_perfect_conducting_wall(u_inner,
                                                    orientation,
                                                    direction,
                                                    x,
                                                    t,
                                                    surface_flux_function,
                                                    equations::MaxwellEquations2D)
    if iseven(direction)
        return surface_flux_function(u_inner,
                                     SVector(-u_inner[1], -u_inner[2], u_inner[3]),
                                     orientation,
                                     equations)
    else
        return surface_flux_function(SVector(-u_inner[1], -u_inner[2], u_inner[3]),
                                     u_inner,
                                     orientation,
                                     equations)
    end
end

function boundary_condition_perfect_conducting_wall(u_inner,
                                                    normal_direction::AbstractVector,
                                                    x,
                                                    t,
                                                    surface_flux_function,
                                                    equations::MaxwellEquations2D)
    return surface_flux_function(u_inner,
                                 SVector(-u_inner[1], -u_inner[2], u_inner[3]),
                                 normal_direction,
                                 equations)
end

function boundary_condition_irradiation(u_inner,
                                        orientation,
                                        direction,
                                        x,
                                        t,
                                        surface_flux_function,
                                        equations::MaxwellEquations2D)
    c = equations.speed_of_light
    k_orth = pi
    k_par = pi
    omega = sqrt(k_orth^2 + k_par^2) * c
    e1 = -(k_orth / k_par) * sin(k_orth * x[2]) * cos(k_par * x[1] - omega * t)
    e2 = cos(k_orth * x[2]) * sin(k_par * x[1] - omega * t)
    if iseven(direction)
        return surface_flux_function(u_inner,
                                     SVector(2.0f0 * e1 - u_inner[1],
                                             2.0f0 * e2 - u_inner[2], u_inner[3]),
                                     orientation,
                                     equations)
    else
        return surface_flux_function(SVector(2.0f0 * e1 - u_inner[1],
                                             2.0f0 * e2 - u_inner[2], u_inner[3]),
                                     u_inner,
                                     orientation,
                                     equations)
    end
end

function boundary_condition_irradiation(u_inner,
                                        normal_direction::AbstractVector,
                                        x,
                                        t,
                                        surface_flux_function,
                                        equations::MaxwellEquations2D)
    c = equations.speed_of_light
    k_orth = pi
    k_par = pi
    omega = sqrt(k_orth^2 + k_par^2) * c
    e1 = -(k_orth / k_par) * sin(k_orth * x[2]) * cos(k_par * x[1] - omega * t)
    e2 = cos(k_orth * x[2]) * sin(k_par * x[1] - omega * t)
    return surface_flux_function(u_inner,
                                 SVector(2.0f0 * e1 - u_inner[1],
                                         2.0f0 * e2 - u_inner[2],
                                         u_inner[3]),
                                 normal_direction,
                                 equations)
end

function boundary_condition_truncation(u_inner,
                                       orientation,
                                       direction,
                                       x,
                                       t,
                                       surface_flux_function,
                                       equations::MaxwellEquations2D)
    if iseven(direction)
        return surface_flux_function(u_inner, SVector(0, 0, 0), orientation, equations)
    else
        return surface_flux_function(SVector(0, 0, 0), u_inner, orientation, equations)
    end
end

function boundary_condition_truncation(u_inner,
                                       normal_direction::AbstractVector,
                                       x,
                                       t,
                                       surface_flux_function,
                                       equations::MaxwellEquations2D)
    return surface_flux_function(u_inner, u_inner, normal_direction, equations)
end

function initial_condition_test(x, t, equations::MaxwellEquations2D)
    c = equations.speed_of_light
    c_sqr = equations.speed_of_light^2
    k_orth = pi
    k_par = pi
    omega = sqrt(k_orth^2 + k_par^2) * c
    e1 = -(k_orth / k_par) * sin(k_orth * x[2]) * cos(k_par * x[1] - omega * t)
    e2 = cos(k_orth * x[2]) * sin(k_par * x[1] - omega * t)
    b = (omega / (k_par * c_sqr)) * cos(k_orth * x[2]) * sin(k_par * x[1] - omega * t)

    return SVector(e1, e2, b)
end

function initial_condition_free_stream_conversion(x, t, equations::MaxwellEquations2D)
    c = equations.speed_of_light
    return SVector(10.0f0, 10.0f0, 10.0f0 / c)
end

function initial_condition_free_stream(x, t, equations::MaxwellEquations2D)
    return SVector(10.0f0, 10.0f0, 10.0f0 / equations.speed_of_light)
end

function initial_condition_convergence(x, t, equations::MaxwellEquations2D)
    c = equations.speed_of_light
    e1 = sin(x[2] + c * t)
    e2 = -sin(x[1] + c * t)
    b = (sin(x[1] + c * t) + sin(x[2] + c * t)) / c

    return SVector(e1, e2, b)
end

function min_max_speed_naive(u_ll, u_rr, orientation, ::MaxwellEquations2D)
    (-equations.speed_of_light, equations.speed_of_light)
end

max_abs_speeds(u, equations::MaxwellEquations2D) = (equations.speed_of_light,
                                        equations.speed_of_light)

max_abs_speed_naive(u_ll, u_rr, orientation, 
                    equations::MaxwellEquations2D) = equations.speed_of_light
end # @muladd
