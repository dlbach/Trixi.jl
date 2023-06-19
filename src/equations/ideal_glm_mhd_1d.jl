# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


    @doc raw"""
        IdealGlmMhdEquations1D(gamma)
    
    The ideal compressible GLM-MHD equations for an ideal gas with ratio of
    specific heats `gamma` in one space dimension.
    
    !!! note
        There is no divergence cleaning variable `psi` because the divergence-free constraint
        is satisfied trivially in one spatial dimension.
    """
    struct IdealGlmMhdEquations1D{RealT<:Real} <: AbstractIdealGlmMhdEquations{1,8}
        gamma::RealT               # ratio of specific heats
        inv_gamma_minus_one::RealT # = inv(gamma - 1); can be used to write slow divisions as fast multiplications

        function IdealGlmMhdEquations1D(gamma)
            γ, inv_gamma_minus_one = promote(gamma, inv(gamma - 1))
            new{typeof(γ)}(γ, inv_gamma_minus_one)
        end
    end

    have_nonconservative_terms(::IdealGlmMhdEquations1D) = False()
    varnames(::typeof(cons2cons), ::IdealGlmMhdEquations1D) =
        ("rho", "rho_v1", "rho_v2", "rho_v3", "rho_e", "B1", "B2", "B3")
    varnames(::typeof(cons2prim), ::IdealGlmMhdEquations1D) =
        ("rho", "v1", "v2", "v3", "p", "B1", "B2", "B3")
    default_analysis_integrals(::IdealGlmMhdEquations1D) =
        (entropy_timederivative, Val(:l2_divb), Val(:linf_divb))


    """
        initial_condition_constant(x, t, equations::IdealGlmMhdEquations1D)
    
    A constant initial condition to test free-stream preservation.
    """
    function initial_condition_constant(x, t, equations::IdealGlmMhdEquations1D)
        rho = 1.0
        rho_v1 = 0.1
        rho_v2 = -0.2
        rho_v3 = -0.5
        rho_e = 50.0
        B1 = 3.0
        B2 = -1.2
        B3 = 0.5
        return SVector(rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3)
    end


    """
        initial_condition_convergence_test(x, t, equations::IdealGlmMhdEquations1D)
    
    An Alfvén wave as smooth initial condition used for convergence tests.
    """
    function initial_condition_convergence_test(x, t, equations::IdealGlmMhdEquations1D)
        # smooth Alfvén wave test from Derigs et al. FLASH (2016)
        # domain must be set to [0, 1], γ = 5/3
        rho = 1.0
        v1 = 0.0
        # TODO: sincospi
        si, co = sincos(2 * pi * x[1])
        v2 = 0.1 * si
        v3 = 0.1 * co
        p = 0.1
        B1 = 1.0
        B2 = v2
        B3 = v3
        return prim2cons(SVector(rho, v1, v2, v3, p, B1, B2, B3), equations)
    end


    """
        initial_condition_weak_blast_wave(x, t, equations::IdealGlmMhdEquations1D)
    
    A weak blast wave adapted from
    - Sebastian Hennemann, Gregor J. Gassner (2020)
      A provably entropy stable subcell shock capturing approach for high order split form DG
      [arXiv: 2008.12044](https://arxiv.org/abs/2008.12044)
    """
    function initial_condition_weak_blast_wave(x, t, equations::IdealGlmMhdEquations1D)
        # Adapted MHD version of the weak blast wave from Hennemann & Gassner JCP paper 2020 (Sec. 6.3)
        # Same discontinuity in the velocities but with magnetic fields
        # Set up polar coordinates
        inicenter = (0,)
        x_norm = x[1] - inicenter[1]
        r = sqrt(x_norm^2)
        phi = atan(x_norm)

        # Calculate primitive variables
        rho = r > 0.5 ? 1.0 : 1.1691
        v1 = r > 0.5 ? 0.0 : 0.1882 * cos(phi)
        p = r > 0.5 ? 1.0 : 1.245

        return prim2cons(SVector(rho, v1, 0.0, 0.0, p, 1.0, 1.0, 1.0, 0.0), equations)
    end


    # Calculate 1D flux in for a single point
    @inline function flux(u, orientation::Integer, equations::IdealGlmMhdEquations1D)
        rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3 = u
        v1 = rho_v1 / rho
        v2 = rho_v2 / rho
        v3 = rho_v3 / rho
        kin_en = 0.5 * (rho_v1 * v1 + rho_v2 * v2 + rho_v3 * v3)
        mag_en = 0.5 * (B1 * B1 + B2 * B2 + B3 * B3)
        p_over_gamma_minus_one = (rho_e - kin_en - mag_en)
        p = (equations.gamma - 1) * p_over_gamma_minus_one

        # Ignore orientation since it is always "1" in 1D
        f1 = rho_v1
        f2 = rho_v1 * v1 + p + mag_en - B1^2
        f3 = rho_v1 * v2 - B1 * B2
        f4 = rho_v1 * v3 - B1 * B3
        f5 =
            (kin_en + equations.gamma * p_over_gamma_minus_one + 2 * mag_en) * v1 -
            B1 * (v1 * B1 + v2 * B2 + v3 * B3)
        f6 = 0.0
        f7 = v1 * B2 - v2 * B1
        f8 = v1 * B3 - v3 * B1

        return SVector(f1, f2, f3, f4, f5, f6, f7, f8)
    end


    """
        flux_derigs_etal(u_ll, u_rr, orientation, equations::IdealGlmMhdEquations1D)
    
    Entropy conserving two-point flux by
    - Derigs et al. (2018)
      Ideal GLM-MHD: About the entropy consistent nine-wave magnetic field
      divergence diminishing ideal magnetohydrodynamics equations
      [DOI: 10.1016/j.jcp.2018.03.002](https://doi.org/10.1016/j.jcp.2018.03.002)
    """
    function flux_derigs_etal(
        u_ll,
        u_rr,
        orientation::Integer,
        equations::IdealGlmMhdEquations1D,
    )
        # Unpack left and right states to get velocities, pressure, and inverse temperature (called beta)
        rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll, B1_ll, B2_ll, B3_ll = u_ll
        rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr, B1_rr, B2_rr, B3_rr = u_rr

        v1_ll = rho_v1_ll / rho_ll
        v2_ll = rho_v2_ll / rho_ll
        v3_ll = rho_v3_ll / rho_ll
        v1_rr = rho_v1_rr / rho_rr
        v2_rr = rho_v2_rr / rho_rr
        v3_rr = rho_v3_rr / rho_rr
        vel_norm_ll = v1_ll^2 + v2_ll^2 + v3_ll^2
        vel_norm_rr = v1_rr^2 + v2_rr^2 + v3_rr^2
        mag_norm_ll = B1_ll^2 + B2_ll^2 + B3_ll^2
        mag_norm_rr = B1_rr^2 + B2_rr^2 + B3_rr^2
        p_ll =
            (equations.gamma - 1) *
            (rho_e_ll - 0.5 * rho_ll * vel_norm_ll - 0.5 * mag_norm_ll)
        p_rr =
            (equations.gamma - 1) *
            (rho_e_rr - 0.5 * rho_rr * vel_norm_rr - 0.5 * mag_norm_rr)
        beta_ll = 0.5 * rho_ll / p_ll
        beta_rr = 0.5 * rho_rr / p_rr
        # for convenience store v⋅B
        vel_dot_mag_ll = v1_ll * B1_ll + v2_ll * B2_ll + v3_ll * B3_ll
        vel_dot_mag_rr = v1_rr * B1_rr + v2_rr * B2_rr + v3_rr * B3_rr

        # Compute the necessary mean values needed for either direction
        rho_avg = 0.5 * (rho_ll + rho_rr)
        rho_mean = ln_mean(rho_ll, rho_rr)
        beta_mean = ln_mean(beta_ll, beta_rr)
        beta_avg = 0.5 * (beta_ll + beta_rr)
        v1_avg = 0.5 * (v1_ll + v1_rr)
        v2_avg = 0.5 * (v2_ll + v2_rr)
        v3_avg = 0.5 * (v3_ll + v3_rr)
        p_mean = 0.5 * rho_avg / beta_avg
        B1_avg = 0.5 * (B1_ll + B1_rr)
        B2_avg = 0.5 * (B2_ll + B2_rr)
        B3_avg = 0.5 * (B3_ll + B3_rr)
        vel_norm_avg = 0.5 * (vel_norm_ll + vel_norm_rr)
        mag_norm_avg = 0.5 * (mag_norm_ll + mag_norm_rr)
        vel_dot_mag_avg = 0.5 * (vel_dot_mag_ll + vel_dot_mag_rr)

        # Ignore orientation since it is always "1" in 1D
        f1 = rho_mean * v1_avg
        f2 = f1 * v1_avg + p_mean + 0.5 * mag_norm_avg - B1_avg * B1_avg
        f3 = f1 * v2_avg - B1_avg * B2_avg
        f4 = f1 * v3_avg - B1_avg * B3_avg
        f6 = 0.0
        f7 = v1_avg * B2_avg - v2_avg * B1_avg
        f8 = v1_avg * B3_avg - v3_avg * B1_avg
        # total energy flux is complicated and involves the previous eight components
        v1_mag_avg = 0.5 * (v1_ll * mag_norm_ll + v1_rr * mag_norm_rr)
        f5 = (
            f1 * 0.5 * (1 / (equations.gamma - 1) / beta_mean - vel_norm_avg) +
            f2 * v1_avg +
            f3 * v2_avg +
            f4 * v3_avg +
            f6 * B1_avg +
            f7 * B2_avg +
            f8 * B3_avg - 0.5 * v1_mag_avg + B1_avg * vel_dot_mag_avg
        )

        return SVector(f1, f2, f3, f4, f5, f6, f7, f8)
    end


    """
        flux_hindenlang_gassner(u_ll, u_rr, orientation_or_normal_direction,
                                equations::IdealGlmMhdEquations1D)
    
    Entropy conserving and kinetic energy preserving two-point flux of
    Hindenlang and Gassner (2019), extending [`flux_ranocha`](@ref) to the MHD equations.
    
    ## References
    - Florian Hindenlang, Gregor Gassner (2019)
      A new entropy conservative two-point flux for ideal MHD equations derived from
      first principles.
      Presented at HONOM 2019: European workshop on high order numerical methods
      for evolutionary PDEs, theory and applications
    - Hendrik Ranocha (2018)
      Generalised Summation-by-Parts Operators and Entropy Stability of Numerical Methods
      for Hyperbolic Balance Laws
      [PhD thesis, TU Braunschweig](https://cuvillier.de/en/shop/publications/7743)
    - Hendrik Ranocha (2020)
      Entropy Conserving and Kinetic Energy Preserving Numerical Methods for
      the Euler Equations Using Summation-by-Parts Operators
      [Proceedings of ICOSAHOM 2018](https://doi.org/10.1007/978-3-030-39647-3_42)
    """
    @inline function flux_hindenlang_gassner(
        u_ll,
        u_rr,
        orientation::Integer,
        equations::IdealGlmMhdEquations1D,
    )
        # Unpack left and right states
        rho_ll, v1_ll, v2_ll, v3_ll, p_ll, B1_ll, B2_ll, B3_ll = cons2prim(u_ll, equations)
        rho_rr, v1_rr, v2_rr, v3_rr, p_rr, B1_rr, B2_rr, B3_rr = cons2prim(u_rr, equations)

        # Compute the necessary mean values needed for either direction
        rho_mean = ln_mean(rho_ll, rho_rr)
        # Algebraically equivalent to `inv_ln_mean(rho_ll / p_ll, rho_rr / p_rr)`
        # in exact arithmetic since
        #     log((ϱₗ/pₗ) / (ϱᵣ/pᵣ)) / (ϱₗ/pₗ - ϱᵣ/pᵣ)
        #   = pₗ pᵣ log((ϱₗ pᵣ) / (ϱᵣ pₗ)) / (ϱₗ pᵣ - ϱᵣ pₗ)
        inv_rho_p_mean = p_ll * p_rr * inv_ln_mean(rho_ll * p_rr, rho_rr * p_ll)
        v1_avg = 0.5 * (v1_ll + v1_rr)
        v2_avg = 0.5 * (v2_ll + v2_rr)
        v3_avg = 0.5 * (v3_ll + v3_rr)
        p_avg = 0.5 * (p_ll + p_rr)
        velocity_square_avg = 0.5 * (v1_ll * v1_rr + v2_ll * v2_rr + v3_ll * v3_rr)
        magnetic_square_avg = 0.5 * (B1_ll * B1_rr + B2_ll * B2_rr + B3_ll * B3_rr)

        # Calculate fluxes depending on orientation with specific direction averages
        f1 = rho_mean * v1_avg
        f2 =
            f1 * v1_avg + p_avg + magnetic_square_avg -
            0.5 * (B1_ll * B1_rr + B1_rr * B1_ll)
        f3 = f1 * v2_avg - 0.5 * (B1_ll * B2_rr + B1_rr * B2_ll)
        f4 = f1 * v3_avg - 0.5 * (B1_ll * B3_rr + B1_rr * B3_ll)
        #f5 below
        f6 = 0.0
        f7 = 0.5 * (v1_ll * B2_ll - v2_ll * B1_ll + v1_rr * B2_rr - v2_rr * B1_rr)
        f8 = 0.5 * (v1_ll * B3_ll - v3_ll * B1_ll + v1_rr * B3_rr - v3_rr * B1_rr)
        # total energy flux is complicated and involves the previous components
        f5 = (
            f1 * (velocity_square_avg + inv_rho_p_mean * equations.inv_gamma_minus_one) +
            0.5 * (
                +p_ll * v1_rr +
                p_rr * v1_ll +
                (v1_ll * B2_ll * B2_rr + v1_rr * B2_rr * B2_ll) +
                (v1_ll * B3_ll * B3_rr + v1_rr * B3_rr * B3_ll) -
                (v2_ll * B1_ll * B2_rr + v2_rr * B1_rr * B2_ll) -
                (v3_ll * B1_ll * B3_rr + v3_rr * B1_rr * B3_ll)
            )
        )

        return SVector(f1, f2, f3, f4, f5, f6, f7, f8)
    end


    # Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
    @inline function max_abs_speed_naive(
        u_ll,
        u_rr,
        orientation::Integer,
        equations::IdealGlmMhdEquations1D,
    )
        rho_ll, rho_v1_ll, _ = u_ll
        rho_rr, rho_v1_rr, _ = u_rr

        # Calculate velocities (ignore orientation since it is always "1" in 1D)
        # and fast magnetoacoustic wave speeds
        # left
        v_ll = rho_v1_ll / rho_ll
        cf_ll = calc_fast_wavespeed(u_ll, orientation, equations)
        # right
        v_rr = rho_v1_rr / rho_rr
        cf_rr = calc_fast_wavespeed(u_rr, orientation, equations)

        λ_max = max(abs(v_ll), abs(v_rr)) + max(cf_ll, cf_rr)
    end


    """
        min_max_speed_naive(u_ll, u_rr, orientation, equations::IdealGlmMhdEquations1D)
    
    Calculate minimum and maximum wave speeds for HLL-type fluxes as in
    - Li (2005)
      An HLLC Riemann solver for magneto-hydrodynamics
      [DOI: 10.1016/j.jcp.2004.08.020](https://doi.org/10.1016/j.jcp.2004.08.020)
    """
    @inline function min_max_speed_naive(
        u_ll,
        u_rr,
        orientation::Integer,
        equations::IdealGlmMhdEquations1D,
    )
        rho_ll, rho_v1_ll, _ = u_ll
        rho_rr, rho_v1_rr, _ = u_rr

        # Calculate primitive variables
        v1_ll = rho_v1_ll / rho_ll
        v1_rr = rho_v1_rr / rho_rr

        # Approximate the left-most and right-most eigenvalues in the Riemann fan
        c_f_ll = calc_fast_wavespeed(u_ll, orientation, equations)
        c_f_rr = calc_fast_wavespeed(u_rr, orientation, equations)
        vel_roe, c_f_roe = calc_fast_wavespeed_roe(u_ll, u_rr, orientation, equations)
        λ_min = min(v1_ll - c_f_ll, vel_roe - c_f_roe)
        λ_max = max(v1_rr + c_f_rr, vel_roe + c_f_roe)

        return λ_min, λ_max
    end


    @inline function max_abs_speeds(u, equations::IdealGlmMhdEquations1D)
        rho, rho_v1, _ = u
        v1 = rho_v1 / rho
        cf_x_direction = calc_fast_wavespeed(u, 1, equations)

        return abs(v1) + cf_x_direction
    end


    # Convert conservative variables to primitive
    @inline function cons2prim(u, equations::IdealGlmMhdEquations1D)
        rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3 = u

        v1 = rho_v1 / rho
        v2 = rho_v2 / rho
        v3 = rho_v3 / rho
        p =
            (equations.gamma - 1) * (
                rho_e -
                0.5 *
                (rho_v1 * v1 + rho_v2 * v2 + rho_v3 * v3 + B1 * B1 + B2 * B2 + B3 * B3)
            )

        return SVector(rho, v1, v2, v3, p, B1, B2, B3)
    end


    # Convert conservative variables to entropy
    @inline function cons2entropy(u, equations::IdealGlmMhdEquations1D)
        rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3 = u

        v1 = rho_v1 / rho
        v2 = rho_v2 / rho
        v3 = rho_v3 / rho
        v_square = v1^2 + v2^2 + v3^2
        p =
            (equations.gamma - 1) *
            (rho_e - 0.5 * rho * v_square - 0.5 * (B1^2 + B2^2 + B3^2))
        s = log(p) - equations.gamma * log(rho)
        rho_p = rho / p

        w1 = (equations.gamma - s) / (equations.gamma - 1) - 0.5 * rho_p * v_square
        w2 = rho_p * v1
        w3 = rho_p * v2
        w4 = rho_p * v3
        w5 = -rho_p
        w6 = rho_p * B1
        w7 = rho_p * B2
        w8 = rho_p * B3

        return SVector(w1, w2, w3, w4, w5, w6, w7, w8)
    end


    # Convert primitive to conservative variables
    @inline function prim2cons(prim, equations::IdealGlmMhdEquations1D)
        rho, v1, v2, v3, p, B1, B2, B3 = prim

        rho_v1 = rho * v1
        rho_v2 = rho * v2
        rho_v3 = rho * v3
        rho_e =
            p / (equations.gamma - 1) +
            0.5 * (rho_v1 * v1 + rho_v2 * v2 + rho_v3 * v3) +
            0.5 * (B1^2 + B2^2 + B3^2)

        return SVector(rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3)
    end


    @inline function density(u, equations::IdealGlmMhdEquations1D)
        rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3 = u
        return rho
    end

    @inline function pressure(u, equations::IdealGlmMhdEquations1D)
        rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3 = u
        p =
            (equations.gamma - 1) * (
                rho_e - 0.5 * (rho_v1^2 + rho_v2^2 + rho_v3^2) / rho -
                0.5 * (B1^2 + B2^2 + B3^2)
            )
        return p
    end

    @inline function density_pressure(u, equations::IdealGlmMhdEquations1D)
        rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3 = u
        p =
            (equations.gamma - 1) * (
                rho_e - 0.5 * (rho_v1^2 + rho_v2^2 + rho_v3^2) / rho -
                0.5 * (B1^2 + B2^2 + B3^2)
            )
        return rho * p
    end


    # Compute the fastest wave speed for ideal MHD equations: c_f, the fast magnetoacoustic eigenvalue
    @inline function calc_fast_wavespeed(cons, direction, equations::IdealGlmMhdEquations1D)
        rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3 = cons
        v1 = rho_v1 / rho
        v2 = rho_v2 / rho
        v3 = rho_v3 / rho
        v_mag = sqrt(v1^2 + v2^2 + v3^2)
        p =
            (equations.gamma - 1) *
            (rho_e - 0.5 * rho * v_mag^2 - 0.5 * (B1^2 + B2^2 + B3^2))
        a_square = equations.gamma * p / rho
        sqrt_rho = sqrt(rho)
        b1 = B1 / sqrt_rho
        b2 = B2 / sqrt_rho
        b3 = B3 / sqrt_rho
        b_square = b1^2 + b2^2 + b3^2

        c_f = sqrt(
            0.5 * (a_square + b_square) +
            0.5 * sqrt((a_square + b_square)^2 - 4.0 * a_square * b1^2),
        )
        return c_f
    end


    """
        calc_fast_wavespeed_roe(u_ll, u_rr, direction, equations::IdealGlmMhdEquations1D)
    
    Compute the fast magnetoacoustic wave speed using Roe averages
    as given by
    - Cargo and Gallice (1997)
      Roe Matrices for Ideal MHD and Systematic Construction
      of Roe Matrices for Systems of Conservation Laws
      [DOI: 10.1006/jcph.1997.5773](https://doi.org/10.1006/jcph.1997.5773)
    """
    @inline function calc_fast_wavespeed_roe(
        u_ll,
        u_rr,
        direction,
        equations::IdealGlmMhdEquations1D,
    )
        rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll, B1_ll, B2_ll, B3_ll = u_ll
        rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr, B1_rr, B2_rr, B3_rr = u_rr

        # Calculate primitive variables
        v1_ll = rho_v1_ll / rho_ll
        v2_ll = rho_v2_ll / rho_ll
        v3_ll = rho_v3_ll / rho_ll
        vel_norm_ll = v1_ll^2 + v2_ll^2 + v3_ll^2
        mag_norm_ll = B1_ll^2 + B2_ll^2 + B3_ll^2
        p_ll =
            (equations.gamma - 1) *
            (rho_e_ll - 0.5 * rho_ll * vel_norm_ll - 0.5 * mag_norm_ll)

        v1_rr = rho_v1_rr / rho_rr
        v2_rr = rho_v2_rr / rho_rr
        v3_rr = rho_v3_rr / rho_rr
        vel_norm_rr = v1_rr^2 + v2_rr^2 + v3_rr^2
        mag_norm_rr = B1_rr^2 + B2_rr^2 + B3_rr^2
        p_rr =
            (equations.gamma - 1) *
            (rho_e_rr - 0.5 * rho_rr * vel_norm_rr - 0.5 * mag_norm_rr)

        # compute total pressure which is thermal + magnetic pressures
        p_total_ll = p_ll + 0.5 * mag_norm_ll
        p_total_rr = p_rr + 0.5 * mag_norm_rr

        # compute the Roe density averages
        sqrt_rho_ll = sqrt(rho_ll)
        sqrt_rho_rr = sqrt(rho_rr)
        inv_sqrt_rho_add = 1.0 / (sqrt_rho_ll + sqrt_rho_rr)
        inv_sqrt_rho_prod = 1.0 / (sqrt_rho_ll * sqrt_rho_rr)
        rho_ll_roe = sqrt_rho_ll * inv_sqrt_rho_add
        rho_rr_roe = sqrt_rho_rr * inv_sqrt_rho_add
        # Roe averages
        # velocities and magnetic fields
        v1_roe = v1_ll * rho_ll_roe + v1_rr * rho_rr_roe
        v2_roe = v2_ll * rho_ll_roe + v2_rr * rho_rr_roe
        v3_roe = v3_ll * rho_ll_roe + v3_rr * rho_rr_roe
        B1_roe = B1_ll * rho_ll_roe + B1_rr * rho_rr_roe
        B2_roe = B2_ll * rho_ll_roe + B2_rr * rho_rr_roe
        B3_roe = B3_ll * rho_ll_roe + B3_rr * rho_rr_roe
        # enthalpy
        H_ll = (rho_e_ll + p_total_ll) / rho_ll
        H_rr = (rho_e_rr + p_total_rr) / rho_rr
        H_roe = H_ll * rho_ll_roe + H_rr * rho_rr_roe
        # temporary variable see equations (4.12) in Cargo and Gallice
        X =
            0.5 *
            ((B1_ll - B1_rr)^2 + (B2_ll - B2_rr)^2 + (B3_ll - B3_rr)^2) *
            inv_sqrt_rho_add^2
        # averaged components needed to compute c_f, the fast magnetoacoustic wave speed
        b_square_roe = (B1_roe^2 + B2_roe^2 + B3_roe^2) * inv_sqrt_rho_prod # scaled magnectic sum
        a_square_roe = (
            (2.0 - equations.gamma) * X +
            (equations.gamma - 1.0) *
            (H_roe - 0.5 * (v1_roe^2 + v2_roe^2 + v3_roe^2) - b_square_roe)
        ) # acoustic speed
        # finally compute the average wave speed and set the output velocity
        # Ignore orientation since it is always "1" in 1D
        c_a_roe = B1_roe^2 * inv_sqrt_rho_prod # (squared) Alfvén wave speed
        a_star_roe = sqrt((a_square_roe + b_square_roe)^2 - 4.0 * a_square_roe * c_a_roe)
        c_f_roe = sqrt(0.5 * (a_square_roe + b_square_roe + a_star_roe))

        return v1_roe, c_f_roe
    end


    # Calculate thermodynamic entropy for a conservative state `cons`
    @inline function entropy_thermodynamic(cons, equations::IdealGlmMhdEquations1D)
        # Pressure
        p =
            (equations.gamma - 1) * (
                cons[5] - 1 / 2 * (cons[2]^2 + cons[3]^2 + cons[4]^2) / cons[1] -
                1 / 2 * (cons[6]^2 + cons[7]^2 + cons[8]^2)
            )

        # Thermodynamic entropy
        s = log(p) - equations.gamma * log(cons[1])

        return s
    end


    # Calculate mathematical entropy for a conservative state `cons`
    @inline function entropy_math(cons, equations::IdealGlmMhdEquations1D)
        S = -entropy_thermodynamic(cons, equations) * cons[1] / (equations.gamma - 1)

        return S
    end


    # Default entropy is the mathematical entropy
    @inline entropy(cons, equations::IdealGlmMhdEquations1D) = entropy_math(cons, equations)


    # Calculate total energy for a conservative state `cons`
    @inline energy_total(cons, ::IdealGlmMhdEquations1D) = cons[5]


    # Calculate kinetic energy for a conservative state `cons`
    @inline function energy_kinetic(cons, equations::IdealGlmMhdEquations1D)
        return 0.5 * (cons[2]^2 + cons[3]^2 + cons[4]^2) / cons[1]
    end


    # Calculate the magnetic energy for a conservative state `cons'.
    #  OBS! For non-dinmensional form of the ideal MHD magnetic pressure ≡ magnetic energy
    @inline function energy_magnetic(cons, ::IdealGlmMhdEquations1D)
        return 0.5 * (cons[6]^2 + cons[7]^2 + cons[8]^2)
    end


    # Calculate internal energy for a conservative state `cons`
    @inline function energy_internal(cons, equations::IdealGlmMhdEquations1D)
        return (
            energy_total(cons, equations) - energy_kinetic(cons, equations) -
            energy_magnetic(cons, equations)
        )
    end


    # Calculate the cross helicity (\vec{v}⋅\vec{B}) for a conservative state `cons'
    @inline function cross_helicity(cons, ::IdealGlmMhdEquations1D)
        return (cons[2] * cons[6] + cons[3] * cons[7] + cons[4] * cons[8]) / cons[1]
    end


end # @muladd
