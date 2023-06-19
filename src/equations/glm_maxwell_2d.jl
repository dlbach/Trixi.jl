# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

    struct GLMMaxwellEquations2D{RealT<:Real} <: AbstractGLMMaxwellEquations{2,4}
        c_h::RealT
        function GLMMaxwellEquations2D(c_h)
            new{typeof(c_h)}(c_h)
        end
    end

    @inline function flux(u, orientation::Integer, equations::GLMMaxwellEquations2D)
        c_sqr = 299792458.0^2
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

    @inline function flux(
        u,
        normal_direction::AbstractVector,
        equations::GLMMaxwellEquations2D,
    )
        c_sqr = 299792458.0^2

        f1 = c_sqr * (normal_direction[1] * u[4] - normal_direction[2] * u[3])
        f2 = c_sqr * (normal_direction[1] * u[3] + normal_direction[2] * u[4])
        f3 = normal_direction[1] * u[2] - normal_direction[2] * u[1]
        f4 = equations.c_h^2 * (normal_direction[1] * u[1] + normal_direction[2] * u[2])

        return SVector(f1, f2, f3, f4)
    end

    @inline function flux_upwind(
        u_ll,
        u_rr,
        orientation::Integer,
        equations::GLMMaxwellEquations2D,
    )
        c = 299792458.0
        u_sum = u_ll + u_rr
        u_diff = u_ll - u_rr
        if orientation == 1
            f1 = 0.5 * c * (equations.c_h * u_diff[1] + c * u_sum[4])
            f2 = 0.5 * c * (u_diff[2] + c * u_sum[3])
            f3 = 0.5 * (u_sum[2] + c * u_diff[3])
            f4 = 0.5 * equations.c_h * (equations.c_h * u_sum[1] + c * u_diff[4])
        else
            f1 = 0.5 * c * (u_diff[1] - c * u_sum[3])
            f2 = 0.5 * c * (equations.c_h * u_diff[2] + c * u_sum[4])
            f3 = 0.5 * (c * u_diff[3] - u_sum[1])
            f4 = 0.5 * equations.c_h * (equations.c_h * u_sum[2] + c * u_diff[4])
        end

        return SVector(f1, f2, f3, f4)
    end


    @inline function flux_upwind(
        u_ll,
        u_rr,
        normal_direction::AbstractVector,
        equations::GLMMaxwellEquations2D,
    )
        c = 299792458.0
        u_sum = u_ll + u_rr
        u_diff = u_ll - u_rr
        flux_component_1 =
            equations.c_h *
            (normal_direction[1] * u_diff[1] + normal_direction[2] * u_diff[2]) +
            c * u_sum[4]
        flux_component_2 =
            normal_direction[1] * u_diff[2] - normal_direction[2] * u_diff[1] + c * u_sum[3]

        f1 =
            0.5 *
            c *
            (
                normal_direction[1] * flux_component_1 -
                normal_direction[2] * flux_component_2
            )
        f2 =
            0.5 *
            c *
            (
                normal_direction[2] * flux_component_1 +
                normal_direction[1] * flux_component_2
            )
        f3 =
            0.5 * (
                normal_direction[1] * u_sum[2] - normal_direction[2] * u_sum[1] +
                c * u_diff[3]
            )
        f4 =
            0.5 *
            equations.c_h *
            (
                equations.c_h *
                (normal_direction[1] * u_sum[1] + normal_direction[2] * u_sum[2]) +
                c * u_diff[4]
            )

        return SVector(f1, f2, f3, f4)
    end

    function boundary_condition_perfect_conducting_wall(
        u_inner,
        orientation,
        direction,
        x,
        t,
        surface_flux_function,
        equations::GLMMaxwellEquations2D,
    )
        c = 299792458.0
        if orientation == 1
            psi_outer = 2.0 * equations.c_h * u_inner[1] / c
        else
            psi_outer = 2.0 * equations.c_h * u_inner[2] / c
        end
        if iseven(direction)
            return surface_flux_function(
                u_inner,
                SVector(-u_inner[1], -u_inner[2], u_inner[3], -u_inner[4] - psi_outer),
                orientation,
                equations,
            )
        else
            return surface_flux_function(
                SVector(-u_inner[1], -u_inner[2], u_inner[3], -u_inner[4] + psi_outer),
                u_inner,
                orientation,
                equations,
            )
        end
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

        c = 299792458.0
        psi_outer =
            2.0 *
            equations.c_h *
            (normal_direction[1] * u_inner[1] + normal_direction[2] * u_inner[2]) / c

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

    varnames(::typeof(cons2cons), ::GLMMaxwellEquations2D) = ("E1", "E2", "B", "psi")
    varnames(::typeof(cons2prim), ::GLMMaxwellEquations2D) = ("E1", "E2", "B", "psi")

    function initial_condition_test(x, t, equations::GLMMaxwellEquations2D)
        c = 299792458.0
        c_sqr = 299792458.0^2
        k_orth = pi
        k_par = pi
        omega = sqrt(k_orth^2 + k_par^2) * c
        e1 = -(k_orth / k_par) * sin(k_orth * x[2]) * cos(k_par * x[1] - omega * t)
        e2 = cos(k_orth * x[2]) * sin(k_par * x[1] - omega * t)
        b = (omega / (k_par * c_sqr)) * cos(k_orth * x[2]) * sin(k_par * x[1] - omega * t)

        return SVector(e1, e2, b, 0.0)
    end

    function initial_condition_free_stream(x, t, equations::GLMMaxwellEquations2D)
        return SVector(10.0, 10.0, 10.0, 10.0)
    end

    function initial_condition_convergence_test(x, t, equations::GLMMaxwellEquations2D)
        c = 299792458.0
        e1 = sin(x[2] + c * t)
        e2 = -sin(x[1] + c * t)
        b = (sin(x[1] + c * t) + sin(x[2] + c * t)) / c

        return SVector(e1, e2, b, 0.0)
    end

    min_max_speed_naive(u_ll, u_rr, orientation, equations::GLMMaxwellEquations2D) =
        max(1.0, equations.c_h) * (-299792458.0, 299792458.0)

    max_abs_speeds(u, equations::GLMMaxwellEquations2D) =
        (max(1.0, equations.c_h) * 299792458.0, max(1.0, equations.c_h) * 299792458.0)

    max_abs_speed_naive(u_ll, u_rr, orientation, equations::GLMMaxwellEquations2D) =
        max(1.0, equations.c_h) * 299792458.0

    # Convert conservative variables to primitive
    @inline cons2prim(u, equations::GLMMaxwellEquations2D) = u

    # Convert conservative variables to entropy variables
    @inline cons2entropy(u, equations::GLMMaxwellEquations2D) = u

end # @muladd
