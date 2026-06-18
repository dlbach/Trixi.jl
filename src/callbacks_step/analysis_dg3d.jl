# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function create_cache_analysis(analyzer, mesh::TreeMesh{3},
                               equations, dg::DG, cache,
                               RealT, uEltype)

    # pre-allocate buffers
    # We use `StrideArray`s here since these buffers are used in performance-critical
    # places and the additional information passed to the compiler makes them faster
    # than native `Array`s.
    u_local = StrideArray(undef, uEltype,
                          StaticInt(nvariables(equations)), StaticInt(nnodes(analyzer)),
                          StaticInt(nnodes(analyzer)), StaticInt(nnodes(analyzer)))
    u_tmp1 = StrideArray(undef, uEltype,
                         StaticInt(nvariables(equations)), StaticInt(nnodes(analyzer)),
                         StaticInt(nnodes(dg)), StaticInt(nnodes(dg)))
    u_tmp2 = StrideArray(undef, uEltype,
                         StaticInt(nvariables(equations)), StaticInt(nnodes(analyzer)),
                         StaticInt(nnodes(analyzer)), StaticInt(nnodes(dg)))
    x_local = StrideArray(undef, RealT,
                          StaticInt(ndims(equations)), StaticInt(nnodes(analyzer)),
                          StaticInt(nnodes(analyzer)), StaticInt(nnodes(analyzer)))
    x_tmp1 = StrideArray(undef, RealT,
                         StaticInt(ndims(equations)), StaticInt(nnodes(analyzer)),
                         StaticInt(nnodes(dg)), StaticInt(nnodes(dg)))
    x_tmp2 = StrideArray(undef, RealT,
                         StaticInt(ndims(equations)), StaticInt(nnodes(analyzer)),
                         StaticInt(nnodes(analyzer)), StaticInt(nnodes(dg)))

    return (; u_local, u_tmp1, u_tmp2, x_local, x_tmp1, x_tmp2)
end

# Specialized cache for P4estMesh to allow for different ambient dimension from mesh dimension
function create_cache_analysis(analyzer,
                               mesh::P4estMesh{3, NDIMS_AMBIENT},
                               equations, dg::DG, cache,
                               RealT, uEltype) where {NDIMS_AMBIENT}

    # pre-allocate buffers
    # We use `StrideArray`s here since these buffers are used in performance-critical
    # places and the additional information passed to the compiler makes them faster
    # than native `Array`s.
    u_local = StrideArray(undef, uEltype,
                          StaticInt(nvariables(equations)), StaticInt(nnodes(analyzer)),
                          StaticInt(nnodes(analyzer)), StaticInt(nnodes(analyzer)))
    u_tmp1 = StrideArray(undef, uEltype,
                         StaticInt(nvariables(equations)), StaticInt(nnodes(analyzer)),
                         StaticInt(nnodes(dg)), StaticInt(nnodes(dg)))
    u_tmp2 = StrideArray(undef, uEltype,
                         StaticInt(nvariables(equations)), StaticInt(nnodes(analyzer)),
                         StaticInt(nnodes(analyzer)), StaticInt(nnodes(dg)))
    x_local = StrideArray(undef, RealT,
                          StaticInt(NDIMS_AMBIENT), StaticInt(nnodes(analyzer)),
                          StaticInt(nnodes(analyzer)), StaticInt(nnodes(analyzer)))
    x_tmp1 = StrideArray(undef, RealT,
                         StaticInt(NDIMS_AMBIENT), StaticInt(nnodes(analyzer)),
                         StaticInt(nnodes(dg)), StaticInt(nnodes(dg)))
    x_tmp2 = StrideArray(undef, RealT,
                         StaticInt(NDIMS_AMBIENT), StaticInt(nnodes(analyzer)),
                         StaticInt(nnodes(analyzer)), StaticInt(nnodes(dg)))
    jacobian_local = StrideArray(undef, RealT,
                                 StaticInt(nnodes(analyzer)),
                                 StaticInt(nnodes(analyzer)),
                                 StaticInt(nnodes(analyzer)))
    jacobian_tmp1 = StrideArray(undef, RealT,
                                StaticInt(nnodes(analyzer)), StaticInt(nnodes(dg)),
                                StaticInt(nnodes(dg)))
    jacobian_tmp2 = StrideArray(undef, RealT,
                                StaticInt(nnodes(analyzer)),
                                StaticInt(nnodes(analyzer)), StaticInt(nnodes(dg)))

    return (; u_local, u_tmp1, u_tmp2, x_local, x_tmp1, x_tmp2, jacobian_local,
            jacobian_tmp1, jacobian_tmp2)
end

function create_cache_analysis(analyzer,
                               mesh::Union{StructuredMesh{3}, T8codeMesh{3}},
                               equations, dg::DG, cache,
                               RealT, uEltype)

    # pre-allocate buffers
    # We use `StrideArray`s here since these buffers are used in performance-critical
    # places and the additional information passed to the compiler makes them faster
    # than native `Array`s.
    u_local = StrideArray(undef, uEltype,
                          StaticInt(nvariables(equations)), StaticInt(nnodes(analyzer)),
                          StaticInt(nnodes(analyzer)), StaticInt(nnodes(analyzer)))
    u_tmp1 = StrideArray(undef, uEltype,
                         StaticInt(nvariables(equations)), StaticInt(nnodes(analyzer)),
                         StaticInt(nnodes(dg)), StaticInt(nnodes(dg)))
    u_tmp2 = StrideArray(undef, uEltype,
                         StaticInt(nvariables(equations)), StaticInt(nnodes(analyzer)),
                         StaticInt(nnodes(analyzer)), StaticInt(nnodes(dg)))
    x_local = StrideArray(undef, RealT,
                          StaticInt(ndims(equations)), StaticInt(nnodes(analyzer)),
                          StaticInt(nnodes(analyzer)), StaticInt(nnodes(analyzer)))
    x_tmp1 = StrideArray(undef, RealT,
                         StaticInt(ndims(equations)), StaticInt(nnodes(analyzer)),
                         StaticInt(nnodes(dg)), StaticInt(nnodes(dg)))
    x_tmp2 = StrideArray(undef, RealT,
                         StaticInt(ndims(equations)), StaticInt(nnodes(analyzer)),
                         StaticInt(nnodes(analyzer)), StaticInt(nnodes(dg)))
    jacobian_local = StrideArray(undef, RealT,
                                 StaticInt(nnodes(analyzer)),
                                 StaticInt(nnodes(analyzer)),
                                 StaticInt(nnodes(analyzer)))
    jacobian_tmp1 = StrideArray(undef, RealT,
                                StaticInt(nnodes(analyzer)), StaticInt(nnodes(dg)),
                                StaticInt(nnodes(dg)))
    jacobian_tmp2 = StrideArray(undef, RealT,
                                StaticInt(nnodes(analyzer)),
                                StaticInt(nnodes(analyzer)), StaticInt(nnodes(dg)))

    return (; u_local, u_tmp1, u_tmp2, x_local, x_tmp1, x_tmp2, jacobian_local,
            jacobian_tmp1, jacobian_tmp2)
end

function calc_error_norms(func, u, t, analyzer,
                          mesh::TreeMesh{3}, equations, initial_condition,
                          dg::DGSEM, cache, cache_analysis)
    @unpack vandermonde, weights = analyzer
    @unpack node_coordinates = cache.elements
    @unpack u_local, u_tmp1, u_tmp2, x_local, x_tmp1, x_tmp2 = cache_analysis

    # Set up data structures
    l2_error = zero(func(get_node_vars(u, equations, dg, 1, 1, 1, 1), equations))
    linf_error = copy(l2_error)

    # Iterate over all elements for error calculations
    for element in eachelement(dg, cache)
        # Interpolate solution and node locations to analysis nodes
        multiply_dimensionwise!(u_local, vandermonde, view(u, :, :, :, :, element),
                                u_tmp1, u_tmp2)
        multiply_dimensionwise!(x_local, vandermonde,
                                view(node_coordinates, :, :, :, :, element), x_tmp1,
                                x_tmp2)

        # Calculate errors at each analysis node
        volume_jacobian_ = volume_jacobian(element, mesh, cache)

        for k in eachnode(analyzer), j in eachnode(analyzer), i in eachnode(analyzer)
            u_exact = initial_condition(get_node_coords(x_local, equations, dg, i, j,
                                                        k), t, equations)
            diff = func(u_exact, equations) -
                   func(get_node_vars(u_local, equations, dg, i, j, k), equations)
            l2_error += diff .^ 2 *
                        (weights[i] * weights[j] * weights[k] * volume_jacobian_)
            linf_error = @. max(linf_error, abs(diff))
        end
    end

    # For L2 error, divide by total volume
    total_volume_ = total_volume(mesh)
    l2_error = @. sqrt(l2_error / total_volume_)

    return l2_error, linf_error
end

function calc_error_norms(func, u, t, analyzer,
                          mesh::Union{StructuredMesh{3}, P4estMesh{3}, T8codeMesh{3}},
                          equations, initial_condition,
                          dg::DGSEM, cache, cache_analysis)
    @unpack vandermonde, weights = analyzer
    @unpack node_coordinates, inverse_jacobian = cache.elements
    @unpack u_local, u_tmp1, u_tmp2, x_local, x_tmp1, x_tmp2, jacobian_local, jacobian_tmp1, jacobian_tmp2 = cache_analysis

    # Set up data structures
    l2_error = zero(func(get_node_vars(u, equations, dg, 1, 1, 1, 1), equations))
    linf_error = copy(l2_error)
    total_volume = zero(real(mesh))

    # Iterate over all elements for error calculations
    for element in eachelement(dg, cache)
        # Interpolate solution and node locations to analysis nodes
        multiply_dimensionwise!(u_local, vandermonde, view(u, :, :, :, :, element),
                                u_tmp1, u_tmp2)
        multiply_dimensionwise!(x_local, vandermonde,
                                view(node_coordinates, :, :, :, :, element), x_tmp1,
                                x_tmp2)
        multiply_scalar_dimensionwise!(jacobian_local, vandermonde,
                                       inv.(view(inverse_jacobian, :, :, :, element)),
                                       jacobian_tmp1, jacobian_tmp2)

        # Calculate errors at each analysis node
        for k in eachnode(analyzer), j in eachnode(analyzer), i in eachnode(analyzer)
            u_exact = initial_condition(get_node_coords(x_local, equations, dg, i, j,
                                                        k), t, equations)
            diff = func(u_exact, equations) -
                   func(get_node_vars(u_local, equations, dg, i, j, k), equations)
            # We take absolute value as we need the Jacobian here for the volume calculation
            abs_jacobian_local_ijk = abs(jacobian_local[i, j, k])

            l2_error += diff .^ 2 *
                        (weights[i] * weights[j] * weights[k] * abs_jacobian_local_ijk)
            linf_error = @. max(linf_error, abs(diff))
            total_volume += (weights[i] * weights[j] * weights[k] *
                             abs_jacobian_local_ijk)
        end
    end

    # For L2 error, divide by total volume
    l2_error = @. sqrt(l2_error / total_volume)

    return l2_error, linf_error
end

function integrate_via_indices(func::Func, u,
                               mesh::TreeMesh{3}, equations, dg::DGSEM, cache,
                               args...; normalize = true) where {Func}
    @unpack weights = dg.basis

    # Initialize integral with zeros of the right shape
    integral = zero(func(u, 1, 1, 1, 1, equations, dg, args...))

    # Use quadrature to numerically integrate over entire domain
    @batch reduction=(+, integral) for element in eachelement(dg, cache)
        volume_jacobian_ = volume_jacobian(element, mesh, cache)
        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            integral += volume_jacobian_ * weights[i] * weights[j] * weights[k] *
                        func(u, i, j, k, element, equations, dg, args...)
        end
    end

    # Normalize with total volume
    if normalize
        integral = integral / total_volume(mesh)
    end

    return integral
end

function integrate_via_indices(func::Func, u,
                               mesh::Union{StructuredMesh{3}, P4estMesh{3},
                                           T8codeMesh{3}},
                               equations, dg::DGSEM, cache,
                               args...; normalize = true) where {Func}
    @unpack weights = dg.basis

    # Initialize integral with zeros of the right shape
    integral = zero(func(u, 1, 1, 1, 1, equations, dg, args...))
    total_volume = zero(real(mesh))

    # Use quadrature to numerically integrate over entire domain
    @batch reduction=((+, integral), (+, total_volume)) for element in eachelement(dg,
                                                                                   cache)
        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            volume_jacobian = abs(inv(cache.elements.inverse_jacobian[i, j, k, element]))
            integral += volume_jacobian * weights[i] * weights[j] * weights[k] *
                        func(u, i, j, k, element, equations, dg, args...)
            total_volume += volume_jacobian * weights[i] * weights[j] * weights[k]
        end
    end

    # Normalize with total volume
    if normalize
        integral = integral / total_volume
    end

    return integral
end

function integrate_interfaces_via_indices(func::Func, u,
                               mesh::TreeMesh{3}, equations, dg::DGSEM, cache,
                               args...; normalize = true) where {Func}
    @unpack weights = dg.basis
    @unpack interfaces = cache

    # Initialize integral with zeros of the right shape
    integral = zero(eltype(u))

    # Use quadrature to numerically integrate over all interfaces
    @batch reduction=(+, integral) for interface in eachinterface(dg, cache)
        element = interfaces.neighbor_ids[1, interface]
        jacobian = inv(cache.elements.inverse_jacobian[element])^2
        for i in eachnode(dg)
            for j in eachnode(dg)
                integral += jacobian * weights[i] * weights[j] * func(u, i, j, interface, equations, dg, cache, args...)
            end
        end
    end

    # Since we integrate a distribution-like value we normalize with the volume of the domain
    # and not with the total area of the interfaces.
    if normalize
        integral = integral / total_volume(mesh)
    end

    return integral
end

function integrate_mortars_via_indices(func::Func, u,
                               mesh::TreeMesh{3}, equations, dg::DGSEM, cache,
                               args...; normalize = true) where {Func}
    @unpack weights = dg.basis
    @unpack interfaces = cache

    # Initialize integral with zeros of the right shape
    integral = zero(eltype(u))

    @batch reduction=(+, integral) for mortar in eachmortar(dg, cache)
        lower_element = cache.mortars.neighbor_ids[1, mortar]
        jacobian = inv(cache.elements.inverse_jacobian[lower_element])^2
        for i in eachnode(dg)
            for j in eachnode(dg)
                integral += jacobian * weights[i] * weights[j] *
                            func(u, i, j, mortar, equations, dg, cache, args...)
            end
        end
    end

    # Normalize with total volume
    if normalize
        integral = integral / total_volume(mesh)
    end

    return integral
end

function integrate(func::Func, u,
                   mesh::Union{TreeMesh{3}, StructuredMesh{3}, P4estMesh{3},
                               T8codeMesh{3}},
                   equations, dg::Union{DGSEM, FDSBP}, cache;
                   normalize = true) where {Func}
    integrate_via_indices(u, mesh, equations, dg, cache;
                          normalize = normalize) do u, i, j, k, element, equations, dg
        u_local = get_node_vars(u, equations, dg, i, j, k, element)
        return func(u_local, equations)
    end
end

function integrate(func::Func, u,
                   mesh::Union{TreeMesh{3}, P4estMesh{3}},
                   equations, equations_parabolic, dg::DGSEM,
                   cache, cache_parabolic; normalize = true) where {Func}
    gradients_x, gradients_y, gradients_z = cache_parabolic.viscous_container.gradients
    integrate_via_indices(u, mesh, equations, dg, cache;
                          normalize = normalize) do u, i, j, k, element, equations, dg
        u_local = get_node_vars(u, equations, dg, i, j, k, element)
        gradients_1_local = get_node_vars(gradients_x, equations_parabolic, dg, i, j, k,
                                          element)
        gradients_2_local = get_node_vars(gradients_y, equations_parabolic, dg, i, j, k,
                                          element)
        gradients_3_local = get_node_vars(gradients_z, equations_parabolic, dg, i, j, k,
                                          element)
        return func(u_local, (gradients_1_local, gradients_2_local, gradients_3_local),
                    equations_parabolic)
    end
end

function analyze(::typeof(entropy_timederivative), du, u, t,
                 mesh::Union{TreeMesh{3}, StructuredMesh{3}, P4estMesh{3},
                             T8codeMesh{3}},
                 equations, dg::Union{DGSEM, FDSBP}, cache)
    # Calculate ∫(∂S/∂u ⋅ ∂u/∂t)dΩ
    integrate_via_indices(u, mesh, equations, dg, cache,
                          du) do u, i, j, k, element, equations, dg, du
        u_node = get_node_vars(u, equations, dg, i, j, k, element)
        du_node = get_node_vars(du, equations, dg, i, j, k, element)
        return dot(cons2entropy(u_node, equations), du_node)
    end
end

function analyze(::Val{:l2_dive}, du, u, t,
                 source_terms, mesh::TreeMesh{3},
                 equations, dg::DGSEM, cache)
    integrate_via_indices(u, mesh, equations, dg, cache, cache,
                          dg.basis.derivative_matrix) do u, i, j, k, element, equations,
                                                         dg, cache, derivative_matrix
        dive = zero(eltype(u))
        for l in eachnode(dg)
            u_ljk = get_node_vars(u, equations, dg, l, j, k, element)
            u_ilk = get_node_vars(u, equations, dg, i, l, k, element)
            u_ijl = get_node_vars(u, equations, dg, i, j, l, element)

            E1_ljk, _, _ = electric_field(u_ljk, equations)
            _, E2_ilk, _ = electric_field(u_ilk, equations)
            _, _, E3_ijl = electric_field(u_ijl, equations)

            dive += (derivative_matrix[i, l] * E1_ljk +
                     derivative_matrix[j, l] * E2_ilk +
                     derivative_matrix[k, l] * E3_ijl)
        end
        u_ijk = get_node_vars(u, equations, dg, i, j, k, element)
        x_ijk = Trixi.get_node_coords(cache.elements.node_coordinates, equations, dg, i, j, k,
                                      element)
        dive = cache.elements.inverse_jacobian[element] * dive - scaled_charge_density(u_ijk, x_ijk, t, source_terms, equations)
        dive^2
    end |> sqrt
end

function analyze(::Val{:l2_dive}, du, u, t,
                 source_terms,
                 mesh::Union{StructuredMesh{3}, UnstructuredMesh2D, P4estMesh{3},
                             T8codeMesh{3}},
                 equations, dg::DGSEM, cache)
    @unpack contravariant_vectors = cache.elements
    integrate_via_indices(u, mesh, equations, dg, cache, cache,
                          dg.basis.derivative_matrix) do u, i, j, element, equations,
                                                         dg, cache, derivative_matrix
        dive = zero(eltype(u))
        # Get the contravariant vectors Ja^1, Ja^2, and Ja^3
        Ja11, Ja12, Ja13 = get_contravariant_vector(1, contravariant_vectors,
                                                    i, j, k, element)
        Ja21, Ja22, Ja23 = get_contravariant_vector(2, contravariant_vectors,
                                                    i, j, k, element)
        Ja31, Ja32, Ja33 = get_contravariant_vector(3, contravariant_vectors,
                                                    i, j, k, element)
        # Compute the transformed divergence
        for l in eachnode(dg)
            u_ljk = get_node_vars(u, equations, dg, l, j, k, element)
            u_ilk = get_node_vars(u, equations, dg, i, l, k, element)
            u_ijl = get_node_vars(u, equations, dg, i, j, l, element)

            E_ljk = magnetic_field(u_ljk, equations)
            E_ilk = magnetic_field(u_ilk, equations)
            E_ijl = magnetic_field(u_ijl, equations)

            dive += (derivative_matrix[i, l] *
                     (Ja11 * E_ljk[1] + Ja12 * E_ljk[2] + Ja13 * E_ljk[3]) +
                     derivative_matrix[j, l] *
                     (Ja21 * E_ilk[1] + Ja22 * E_ilk[2] + Ja23 * E_ilk[3]) +
                     derivative_matrix[k, l] *
                     (Ja31 * E_ijl[1] + Ja32 * E_ijl[2] + Ja33 * E_ijl[3]))
        end

        u_ijk = get_node_vars(u, equations, dg, i, j, k, element)
        x_ijk = Trixi.get_node_coords(cache.elements.node_coordinates, equations, dg, i, j, k,
                                      element)
        dive = cache.elements.inverse_jacobian[i, j, k, element] * dive - scaled_charge_density(u_ijk, x_ijk, t, source_terms, equations)
        dive^2
    end |> sqrt
end

function analyze(::Val{:l2_e_normal_jump}, du, u, t,
                 mesh::TreeMesh{3},
                 equations, dg::DGSEM, cache)
    a = integrate_interfaces_via_indices(u, mesh, equations, dg, cache) do u, i, j, interface, equations, dg, cache                                                
        @unpack u, orientations = cache.interfaces
        normal_jump = zero(eltype(u))
        u_ll, u_rr = get_surface_node_vars(u, equations, dg, i, j, interface)
        if orientations[interface] == 1
            E1_ll, _, _ = electric_field(u_ll, equations)
            E1_rr, _, _ = electric_field(u_rr, equations)
            normal_jump += (E1_ll - E1_rr)^2
        elseif orientations[interface] == 2
            _, E2_ll, _ = electric_field(u_ll, equations)
            _, E2_rr, _ = electric_field(u_rr, equations)
            normal_jump += (E2_ll - E2_rr)^2
        else
            _, _, E3_ll = electric_field(u_ll, equations)
            _, _, E3_rr = electric_field(u_rr, equations)
            normal_jump += (E3_ll - E3_rr)^2
        end
    end

    b = integrate_mortars_via_indices(u, mesh, equations, dg, cache) do u, i, j, mortar, equations, dg, cache                                                        
        @unpack u_upper_left, u_upper_right, u_lower_left, u_lower_right, orientations = cache.mortars
        normal_jump = zero(eltype(u))
        u_ul_ll, u_ul_rr = get_surface_node_vars(u_upper_left, equations, dg,
                                                        i, j, mortar)
        u_ur_ll, u_ur_rr = get_surface_node_vars(u_upper_right, equations, dg,
                                                        i, j, mortar)
        u_ll_ll, u_ll_rr = get_surface_node_vars(u_lower_left, equations, dg,
                                                        i, j, mortar)
        u_lr_ll, u_lr_rr = get_surface_node_vars(u_lower_right, equations, dg,
                                                        i, j, mortar)   

        if orientations[mortar] == 1
            E1_ul_ll, _, _ = electric_field(u_ul_ll, equations)
            E1_ul_rr, _, _ = electric_field(u_ul_rr, equations)
            E1_ur_ll, _, _ = electric_field(u_ur_ll, equations)
            E1_ur_rr, _, _ = electric_field(u_ur_rr, equations)
            E1_ll_ll, _, _ = electric_field(u_ll_ll, equations)
            E1_ll_rr, _, _ = electric_field(u_ll_rr, equations)
            E1_lr_ll, _, _ = electric_field(u_lr_ll, equations)
            E1_lr_rr, _, _ = electric_field(u_lr_rr, equations)
            normal_jump += (E1_ul_ll - E1_ul_rr)^2 + (E1_ur_ll - E1_ur_rr)^2 +
                           (E1_ll_ll - E1_ll_rr)^2 + (E1_lr_ll - E1_lr_rr)^2
        elseif orientations[mortar] == 2
            _, E2_ul_ll, _ = electric_field(u_ul_ll, equations)
            _, E2_ul_rr, _ = electric_field(u_ul_rr, equations)
            _, E2_ur_ll, _ = electric_field(u_ur_ll, equations)
            _, E2_ur_rr, _ = electric_field(u_ur_rr, equations)
            _, E2_ll_ll, _ = electric_field(u_ll_ll, equations)
            _, E2_ll_rr, _ = electric_field(u_ll_rr, equations)
            _, E2_lr_ll, _ = electric_field(u_lr_ll, equations)
            _, E2_lr_rr, _ = electric_field(u_lr_rr, equations)
            normal_jump += (E2_ul_ll - E2_ul_rr)^2 + (E2_ur_ll - E2_ur_rr)^2 +
                           (E2_ll_ll - E2_ll_rr)^2 + (E2_lr_ll - E2_lr_rr)^2
        else
            _, _, E3_ul_ll = electric_field(u_ul_ll, equations)
            _, _, E3_ul_rr = electric_field(u_ul_rr, equations)
            _, _, E3_ur_ll = electric_field(u_ur_ll, equations)
            _, _, E3_ur_rr = electric_field(u_ur_rr, equations)
            _, _, E3_ll_ll = electric_field(u_ll_ll, equations)
            _, _, E3_ll_rr = electric_field(u_ll_rr, equations)
            _, _, E3_lr_ll = electric_field(u_lr_ll, equations)
            _, _, E3_lr_rr = electric_field(u_lr_rr, equations)
            normal_jump += (E3_ul_ll - E3_ul_rr)^2 + (E3_ur_ll - E3_ur_rr)^2 +
                           (E3_ll_ll - E3_ll_rr)^2 + (E3_lr_ll - E3_lr_rr)^2
        end
    end

    return sqrt(a + b)
end

function analyze(::Val{:l2_divb}, du, u, t,
                 mesh::TreeMesh{3}, equations,
                 dg::DGSEM, cache)
    integrate_via_indices(u, mesh, equations, dg, cache, cache,
                          dg.basis.derivative_matrix) do u, i, j, k, element, equations,
                                                         dg, cache, derivative_matrix
        divb = zero(eltype(u))
        for l in eachnode(dg)
            u_ljk = get_node_vars(u, equations, dg, l, j, k, element)
            u_ilk = get_node_vars(u, equations, dg, i, l, k, element)
            u_ijl = get_node_vars(u, equations, dg, i, j, l, element)

            B_ljk = magnetic_field(u_ljk, equations)
            B_ilk = magnetic_field(u_ilk, equations)
            B_ijl = magnetic_field(u_ijl, equations)

            divb += (derivative_matrix[i, l] * B_ljk[1] +
                     derivative_matrix[j, l] * B_ilk[2] +
                     derivative_matrix[k, l] * B_ijl[3])
        end
        divb *= cache.elements.inverse_jacobian[element]
        return divb^2
    end |> sqrt
end

function analyze(::Val{:l2_divb}, du, u, t,
                 mesh::Union{StructuredMesh{3}, P4estMesh{3}, T8codeMesh{3}},
                 equations,
                 dg::DGSEM, cache)
    @unpack contravariant_vectors = cache.elements
    integrate_via_indices(u, mesh, equations, dg, cache, cache,
                          dg.basis.derivative_matrix) do u, i, j, k, element, equations,
                                                         dg, cache, derivative_matrix
        divb = zero(eltype(u))
        # Get the contravariant vectors Ja^1, Ja^2, and Ja^3
        Ja11, Ja12, Ja13 = get_contravariant_vector(1, contravariant_vectors,
                                                    i, j, k, element)
        Ja21, Ja22, Ja23 = get_contravariant_vector(2, contravariant_vectors,
                                                    i, j, k, element)
        Ja31, Ja32, Ja33 = get_contravariant_vector(3, contravariant_vectors,
                                                    i, j, k, element)
        # Compute the transformed divergence
        for l in eachnode(dg)
            u_ljk = get_node_vars(u, equations, dg, l, j, k, element)
            u_ilk = get_node_vars(u, equations, dg, i, l, k, element)
            u_ijl = get_node_vars(u, equations, dg, i, j, l, element)

            B_ljk = magnetic_field(u_ljk, equations)
            B_ilk = magnetic_field(u_ilk, equations)
            B_ijl = magnetic_field(u_ijl, equations)

            divb += (derivative_matrix[i, l] *
                     (Ja11 * B_ljk[1] + Ja12 * B_ljk[2] + Ja13 * B_ljk[3]) +
                     derivative_matrix[j, l] *
                     (Ja21 * B_ilk[1] + Ja22 * B_ilk[2] + Ja23 * B_ilk[3]) +
                     derivative_matrix[k, l] *
                     (Ja31 * B_ijl[1] + Ja32 * B_ijl[2] + Ja33 * B_ijl[3]))
        end
        divb *= cache.elements.inverse_jacobian[i, j, k, element]
        return divb^2
    end |> sqrt
end

function analyze(::Val{:linf_divb}, du, u, t,
                 mesh::TreeMesh{3}, equations,
                 dg::DGSEM, cache)
    @unpack derivative_matrix, weights = dg.basis

    # integrate over all elements to get the divergence-free condition errors
    linf_divb = zero(eltype(u))
    @batch reduction=(max, linf_divb) for element in eachelement(dg, cache)
        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            divb = zero(eltype(u))
            for l in eachnode(dg)
                u_ljk = get_node_vars(u, equations, dg, l, j, k, element)
                u_ilk = get_node_vars(u, equations, dg, i, l, k, element)
                u_ijl = get_node_vars(u, equations, dg, i, j, l, element)

                B_ljk = magnetic_field(u_ljk, equations)
                B_ilk = magnetic_field(u_ilk, equations)
                B_ijl = magnetic_field(u_ijl, equations)

                divb += (derivative_matrix[i, l] * B_ljk[1] +
                         derivative_matrix[j, l] * B_ilk[2] +
                         derivative_matrix[k, l] * B_ijl[3])
            end
            divb *= cache.elements.inverse_jacobian[element]
            linf_divb = max(linf_divb, abs(divb))
        end
    end

    if mpi_isparallel()
        # Base.max instead of max needed, see comment in src/auxiliary/math.jl
        linf_divb = MPI.Allreduce!(Ref(linf_divb), Base.max, mpi_comm())[]
    end

    return linf_divb
end

function analyze(::Val{:linf_divb}, du, u, t,
                 mesh::Union{StructuredMesh{3}, P4estMesh{3}, T8codeMesh{3}},
                 equations,
                 dg::DGSEM, cache)
    @unpack derivative_matrix, weights = dg.basis
    @unpack contravariant_vectors = cache.elements

    # integrate over all elements to get the divergence-free condition errors
    linf_divb = zero(eltype(u))
    @batch reduction=(max, linf_divb) for element in eachelement(dg, cache)
        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            divb = zero(eltype(u))
            # Get the contravariant vectors Ja^1, Ja^2, and Ja^3
            Ja11, Ja12, Ja13 = get_contravariant_vector(1, contravariant_vectors,
                                                        i, j, k, element)
            Ja21, Ja22, Ja23 = get_contravariant_vector(2, contravariant_vectors,
                                                        i, j, k, element)
            Ja31, Ja32, Ja33 = get_contravariant_vector(3, contravariant_vectors,
                                                        i, j, k, element)
            # Compute the transformed divergence
            for l in eachnode(dg)
                u_ljk = get_node_vars(u, equations, dg, l, j, k, element)
                u_ilk = get_node_vars(u, equations, dg, i, l, k, element)
                u_ijl = get_node_vars(u, equations, dg, i, j, l, element)

                B_ljk = magnetic_field(u_ljk, equations)
                B_ilk = magnetic_field(u_ilk, equations)
                B_ijl = magnetic_field(u_ijl, equations)

                divb += (derivative_matrix[i, l] * (Ja11 * B_ljk[1] +
                          Ja12 * B_ljk[2] + Ja13 * B_ljk[3]) +
                         derivative_matrix[j, l] * (Ja21 * B_ilk[1] +
                          Ja22 * B_ilk[2] + Ja23 * B_ilk[3]) +
                         derivative_matrix[k, l] * (Ja31 * B_ijl[1] +
                          Ja32 * B_ijl[2] + Ja33 * B_ijl[3]))
            end
            divb *= cache.elements.inverse_jacobian[i, j, k, element]
            linf_divb = max(linf_divb, abs(divb))
        end
    end

    if mpi_isparallel()
        # Base.max instead of max needed, see comment in src/auxiliary/math.jl
        linf_divb = MPI.Allreduce!(Ref(linf_divb), Base.max, mpi_comm())[]
    end

    return linf_divb
end

function analyze(::Val{:l2_b_normal_jump}, du, u, t,
                 mesh::TreeMesh{3},
                 equations, dg::DGSEM, cache)
    a = integrate_interfaces_via_indices(u, mesh, equations, dg, cache) do u, i, j, interface, equations, dg, cache                                                
        @unpack u, orientations = cache.interfaces
        normal_jump = zero(eltype(u))
        u_ll, u_rr = get_surface_node_vars(u, equations, dg, i, j, interface)
        if orientations[interface] == 1
            B1_ll, _, _ = magnetic_field(u_ll, equations)
            B1_rr, _, _ = magnetic_field(u_rr, equations)
            normal_jump += (B1_ll - B1_rr)^2
        elseif orientations[interface] == 2
            _, B2_ll, _ = magnetic_field(u_ll, equations)
            _, B2_rr, _ = magnetic_field(u_rr, equations)
            normal_jump += (B2_ll - B2_rr)^2
        else
            _, _, B3_ll = magnetic_field(u_ll, equations)
            _, _, B3_rr = magnetic_field(u_rr, equations)
            normal_jump += (B3_ll - B3_rr)^2
        end
    end

    b = integrate_mortars_via_indices(u, mesh, equations, dg, cache) do u, i, j, mortar, equations, dg, cache                                                        
        @unpack u_upper_left, u_upper_right, u_lower_left, u_lower_right, orientations = cache.mortars
        normal_jump = zero(eltype(u))
        u_ul_ll, u_ul_rr = get_surface_node_vars(u_upper_left, equations, dg,
                                                        i, j, mortar)
        u_ur_ll, u_ur_rr = get_surface_node_vars(u_upper_right, equations, dg,
                                                        i, j, mortar)
        u_ll_ll, u_ll_rr = get_surface_node_vars(u_lower_left, equations, dg,
                                                        i, j, mortar)
        u_lr_ll, u_lr_rr = get_surface_node_vars(u_lower_right, equations, dg,
                                                        i, j, mortar)   

        if orientations[mortar] == 1
            B1_ul_ll, _, _ = magnetic_field(u_ul_ll, equations)
            B1_ul_rr, _, _ = magnetic_field(u_ul_rr, equations)
            B1_ur_ll, _, _ = magnetic_field(u_ur_ll, equations)
            B1_ur_rr, _, _ = magnetic_field(u_ur_rr, equations)
            B1_ll_ll, _, _ = magnetic_field(u_ll_ll, equations)
            B1_ll_rr, _, _ = magnetic_field(u_ll_rr, equations)
            B1_lr_ll, _, _ = magnetic_field(u_lr_ll, equations)
            B1_lr_rr, _, _ = magnetic_field(u_lr_rr, equations)
            normal_jump += (B1_ul_ll - B1_ul_rr)^2 + (B1_ur_ll - B1_ur_rr)^2 +
                           (B1_ll_ll - B1_ll_rr)^2 + (B1_lr_ll - B1_lr_rr)^2
        elseif orientations[mortar] == 2
            _, B2_ul_ll, _ = magnetic_field(u_ul_ll, equations)
            _, B2_ul_rr, _ = magnetic_field(u_ul_rr, equations)
            _, B2_ur_ll, _ = magnetic_field(u_ur_ll, equations)
            _, B2_ur_rr, _ = magnetic_field(u_ur_rr, equations)
            _, B2_ll_ll, _ = magnetic_field(u_ll_ll, equations)
            _, B2_ll_rr, _ = magnetic_field(u_ll_rr, equations)
            _, B2_lr_ll, _ = magnetic_field(u_lr_ll, equations)
            _, B2_lr_rr, _ = magnetic_field(u_lr_rr, equations)
            normal_jump += (B2_ul_ll - B2_ul_rr)^2 + (B2_ur_ll - B2_ur_rr)^2 +
                           (B2_ll_ll - B2_ll_rr)^2 + (B2_lr_ll - B2_lr_rr)^2
        else
            _, _, B3_ul_ll = magnetic_field(u_ul_ll, equations)
            _, _, B3_ul_rr = magnetic_field(u_ul_rr, equations)
            _, _, B3_ur_ll = magnetic_field(u_ur_ll, equations)
            _, _, B3_ur_rr = magnetic_field(u_ur_rr, equations)
            _, _, B3_ll_ll = magnetic_field(u_ll_ll, equations)
            _, _, B3_ll_rr = magnetic_field(u_ll_rr, equations)
            _, _, B3_lr_ll = magnetic_field(u_lr_ll, equations)
            _, _, B3_lr_rr = magnetic_field(u_lr_rr, equations)
            normal_jump += (B3_ul_ll - B3_ul_rr)^2 + (B3_ur_ll - B3_ur_rr)^2 +
                           (B3_ll_ll - B3_ll_rr)^2 + (B3_lr_ll - B3_lr_rr)^2
        end
    end

    return sqrt(a + b)
end

end # @muladd
