# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function create_cache_analysis(analyzer, mesh::TreeMesh{2},
                               equations, dg::DG, cache,
                               RealT, uEltype)

    # pre-allocate buffers
    # We use `StrideArray`s here since these buffers are used in performance-critical
    # places and the additional information passed to the compiler makes them faster
    # than native `Array`s.
    u_local = StrideArray(undef, uEltype,
                          StaticInt(nvariables(equations)), StaticInt(nnodes(analyzer)),
                          StaticInt(nnodes(analyzer)))
    u_tmp1 = StrideArray(undef, uEltype,
                         StaticInt(nvariables(equations)), StaticInt(nnodes(analyzer)),
                         StaticInt(nnodes(dg)))
    x_local = StrideArray(undef, RealT,
                          StaticInt(ndims(equations)), StaticInt(nnodes(analyzer)),
                          StaticInt(nnodes(analyzer)))
    x_tmp1 = StrideArray(undef, RealT,
                         StaticInt(ndims(equations)), StaticInt(nnodes(analyzer)),
                         StaticInt(nnodes(dg)))

    return (; u_local, u_tmp1, x_local, x_tmp1)
end

# Specialized cache for P4estMesh to allow for different ambient dimension from mesh dimension
function create_cache_analysis(analyzer,
                               mesh::Union{P4estMesh{2, NDIMS_AMBIENT},
                                           P4estMeshView{2, NDIMS_AMBIENT}},
                               equations, dg::DG, cache,
                               RealT, uEltype) where {NDIMS_AMBIENT}

    # pre-allocate buffers
    # We use `StrideArray`s here since these buffers are used in performance-critical
    # places and the additional information passed to the compiler makes them faster
    # than native `Array`s.
    u_local = StrideArray(undef, uEltype,
                          StaticInt(nvariables(equations)), StaticInt(nnodes(analyzer)),
                          StaticInt(nnodes(analyzer)))
    u_tmp1 = StrideArray(undef, uEltype,
                         StaticInt(nvariables(equations)), StaticInt(nnodes(analyzer)),
                         StaticInt(nnodes(dg)))
    x_local = StrideArray(undef, RealT,
                          StaticInt(NDIMS_AMBIENT), StaticInt(nnodes(analyzer)),
                          StaticInt(nnodes(analyzer)))
    x_tmp1 = StrideArray(undef, RealT,
                         StaticInt(NDIMS_AMBIENT), StaticInt(nnodes(analyzer)),
                         StaticInt(nnodes(dg)))
    jacobian_local = StrideArray(undef, RealT,
                                 StaticInt(nnodes(analyzer)),
                                 StaticInt(nnodes(analyzer)))
    jacobian_tmp1 = StrideArray(undef, RealT,
                                StaticInt(nnodes(analyzer)), StaticInt(nnodes(dg)))

    return (; u_local, u_tmp1, x_local, x_tmp1, jacobian_local, jacobian_tmp1)
end

function create_cache_analysis(analyzer,
                               mesh::Union{StructuredMesh{2}, StructuredMeshView{2},
                                           UnstructuredMesh2D, T8codeMesh{2}},
                               equations, dg::DG, cache,
                               RealT, uEltype)

    # pre-allocate buffers
    # We use `StrideArray`s here since these buffers are used in performance-critical
    # places and the additional information passed to the compiler makes them faster
    # than native `Array`s.
    u_local = StrideArray(undef, uEltype,
                          StaticInt(nvariables(equations)), StaticInt(nnodes(analyzer)),
                          StaticInt(nnodes(analyzer)))
    u_tmp1 = StrideArray(undef, uEltype,
                         StaticInt(nvariables(equations)), StaticInt(nnodes(analyzer)),
                         StaticInt(nnodes(dg)))
    x_local = StrideArray(undef, RealT,
                          StaticInt(ndims(equations)), StaticInt(nnodes(analyzer)),
                          StaticInt(nnodes(analyzer)))
    x_tmp1 = StrideArray(undef, RealT,
                         StaticInt(ndims(equations)), StaticInt(nnodes(analyzer)),
                         StaticInt(nnodes(dg)))
    jacobian_local = StrideArray(undef, RealT,
                                 StaticInt(nnodes(analyzer)),
                                 StaticInt(nnodes(analyzer)))
    jacobian_tmp1 = StrideArray(undef, RealT,
                                StaticInt(nnodes(analyzer)), StaticInt(nnodes(dg)))

    return (; u_local, u_tmp1, x_local, x_tmp1, jacobian_local, jacobian_tmp1)
end

function calc_error_norms(func, u, t, analyzer,
                          mesh::TreeMesh{2}, equations, initial_condition,
                          dg::DGSEM, cache, cache_analysis)
    @unpack vandermonde, weights = analyzer
    @unpack node_coordinates = cache.elements
    @unpack u_local, u_tmp1, x_local, x_tmp1 = cache_analysis

    # Set up data structures
    l2_error = zero(func(get_node_vars(u, equations, dg, 1, 1, 1), equations))
    linf_error = copy(l2_error)

    # Iterate over all elements for error calculations
    # Accumulate L2 error on the element first so that the order of summation is the
    # same as in the parallel case to ensure exact equality. This facilitates easier parallel
    # development and debugging (see
    # https://github.com/trixi-framework/Trixi.jl/pull/850#pullrequestreview-757463943 for details).
    for element in eachelement(dg, cache)
        # Set up data structures for local element L2 error
        l2_error_local = zero(l2_error)

        # Interpolate solution and node locations to analysis nodes
        multiply_dimensionwise!(u_local, vandermonde, view(u, :, :, :, element), u_tmp1)
        multiply_dimensionwise!(x_local, vandermonde,
                                view(node_coordinates, :, :, :, element), x_tmp1)

        # Calculate errors at each analysis node
        volume_jacobian_ = volume_jacobian(element, mesh, cache)

        for j in eachnode(analyzer), i in eachnode(analyzer)
            u_exact = initial_condition(get_node_coords(x_local, equations, dg, i, j),
                                        t, equations)
            diff = func(u_exact, equations) -
                   func(get_node_vars(u_local, equations, dg, i, j), equations)
            l2_error_local += diff .^ 2 * (weights[i] * weights[j] * volume_jacobian_)
            linf_error = @. max(linf_error, abs(diff))
        end
        l2_error += l2_error_local
    end

    # For L2 error, divide by total volume
    total_volume_ = total_volume(mesh)
    l2_error = @. sqrt(l2_error / total_volume_)

    return l2_error, linf_error
end

function calc_error_norms(func, u, t, analyzer,
                          mesh::Union{StructuredMesh{2}, StructuredMeshView{2},
                                      UnstructuredMesh2D,
                                      P4estMesh{2}, P4estMeshView{2},
                                      T8codeMesh{2}},
                          equations,
                          initial_condition, dg::DGSEM, cache, cache_analysis)
    @unpack vandermonde, weights = analyzer
    @unpack node_coordinates, inverse_jacobian = cache.elements
    @unpack u_local, u_tmp1, x_local, x_tmp1, jacobian_local, jacobian_tmp1 = cache_analysis

    # Set up data structures
    l2_error = zero(func(get_node_vars(u, equations, dg, 1, 1, 1), equations))
    linf_error = copy(l2_error)
    total_volume = zero(real(mesh))

    # Iterate over all elements for error calculations
    for element in eachelement(dg, cache)
        # Interpolate solution and node locations to analysis nodes
        multiply_dimensionwise!(u_local, vandermonde, view(u, :, :, :, element), u_tmp1)
        multiply_dimensionwise!(x_local, vandermonde,
                                view(node_coordinates, :, :, :, element), x_tmp1)
        multiply_scalar_dimensionwise!(jacobian_local, vandermonde,
                                       inv.(view(inverse_jacobian, :, :, element)),
                                       jacobian_tmp1)

        # Calculate errors at each analysis node
        for j in eachnode(analyzer), i in eachnode(analyzer)
            u_exact = initial_condition(get_node_coords(x_local, equations, dg, i, j),
                                        t, equations)
            diff = func(u_exact, equations) -
                   func(get_node_vars(u_local, equations, dg, i, j), equations)
            # We take absolute value as we need the Jacobian here for the volume calculation
            abs_jacobian_local_ij = abs(jacobian_local[i, j])

            l2_error += diff .^ 2 * (weights[i] * weights[j] * abs_jacobian_local_ij)
            linf_error = @. max(linf_error, abs(diff))
            total_volume += weights[i] * weights[j] * abs_jacobian_local_ij
        end
    end

    # For L2 error, divide by total volume
    l2_error = @. sqrt(l2_error / total_volume)

    return l2_error, linf_error
end

function integrate_via_indices(func::Func, u,
                               mesh::TreeMesh{2}, equations, dg::DGSEM, cache,
                               args...; normalize = true) where {Func}
    @unpack weights = dg.basis

    # Initialize integral with zeros of the right shape
    integral = zero(func(u, 1, 1, 1, equations, dg, args...))

    # Use quadrature to numerically integrate over entire domain
    @batch reduction=(+, integral) for element in eachelement(dg, cache)
        jacobian_ = volume_jacobian(element, mesh, cache)
        for j in eachnode(dg), i in eachnode(dg)
            integral += jacobian_ * weights[i] * weights[j] *
                        func(u, i, j, element, equations, dg, args...)
        end
    end

    # Normalize with total volume
    if normalize
        integral = integral / total_volume(mesh)
    end

    return integral
end

function integrate_via_indices(func::Func, u,
                               mesh::Union{StructuredMesh{2}, StructuredMeshView{2},
                                           UnstructuredMesh2D, P4estMesh{2},
                                           T8codeMesh{2}},
                               equations,
                               dg::DGSEM, cache, args...; normalize = true) where {Func}
    @unpack weights = dg.basis

    # Initialize integral with zeros of the right shape
    integral = zero(func(u, 1, 1, 1, equations, dg, args...))
    total_volume = zero(real(mesh))

    # Use quadrature to numerically integrate over entire domain
    @batch reduction=((+, integral), (+, total_volume)) for element in eachelement(dg,
                                                                                   cache)
        for j in eachnode(dg), i in eachnode(dg)
            volume_jacobian = abs(inv(cache.elements.inverse_jacobian[i, j, element]))
            integral += volume_jacobian * weights[i] * weights[j] *
                        func(u, i, j, element, equations, dg, args...)
            total_volume += volume_jacobian * weights[i] * weights[j]
        end
    end

    # Normalize with total volume
    if normalize
        integral = integral / total_volume
    end

    return integral
end

function integrate_interfaces_via_indices(func::Func, u,
                               mesh::TreeMesh{2}, equations, dg::DGSEM, cache,
                               args...; normalize = true) where {Func}
    @unpack weights = dg.basis
    @unpack interfaces = cache

    # Initialize integral with zeros of the right shape
    integral = zero(eltype(u))

    # Use quadrature to numerically integrate over entire domain
    @batch reduction=(+, integral) for interface in eachinterface(dg, cache)
        element = interfaces.neighbor_ids[1, interface]
        jacobian = inv(cache.elements.inverse_jacobian[element])
        for i in eachnode(dg)
            integral += jacobian * weights[i] * func(u, i, interface, equations, dg, args...)
        end
    end

    # Normalize with total volume
    if normalize
        integral = integral / total_volume(mesh)
    end

    return integral
end

function integrate_mortars_via_indices(func::Func, u,
                               mesh::TreeMesh{2}, equations, dg::DGSEM, cache,
                               args...; normalize = true) where {Func}
    @unpack weights = dg.basis
    @unpack interfaces = cache

    # Initialize integral with zeros of the right shape
    integral = zero(eltype(u))

    @batch reduction=(+, integral) for mortar in eachmortar(dg, cache)
        lower_element = cache.mortars.neighbor_ids[1, mortar]
        jacobian = inv(cache.elements.inverse_jacobian[lower_element])
        for i in eachnode(dg)
            integral += jacobian * weights[i] *
                        func(u, i, mortar, equations, dg, args...)
        end
    end

    # Normalize with total volume
    if normalize
        integral = integral / total_volume(mesh)
    end

    return integral
end


function integrate(func::Func, u,
                   mesh::Union{TreeMesh{2}, StructuredMesh{2}, StructuredMeshView{2},
                               UnstructuredMesh2D, P4estMesh{2}, P4estMeshView{2},
                               T8codeMesh{2}},
                   equations, dg::Union{DGSEM, FDSBP}, cache;
                   normalize = true) where {Func}
    integrate_via_indices(u, mesh, equations, dg, cache;
                          normalize = normalize) do u, i, j, element, equations, dg
        u_local = get_node_vars(u, equations, dg, i, j, element)
        return func(u_local, equations)
    end
end

function integrate(func::Func, u,
                   mesh::Union{TreeMesh{2}, P4estMesh{2}},
                   equations, equations_parabolic, dg::DGSEM,
                   cache, cache_parabolic; normalize = true) where {Func}
    gradients_x, gradients_y = cache_parabolic.viscous_container.gradients
    integrate_via_indices(u, mesh, equations, dg, cache;
                          normalize = normalize) do u, i, j, element, equations, dg
        u_local = get_node_vars(u, equations, dg, i, j, element)
        gradients_1_local = get_node_vars(gradients_x, equations_parabolic, dg, i, j,
                                          element)
        gradients_2_local = get_node_vars(gradients_y, equations_parabolic, dg, i, j,
                                          element)
        return func(u_local, (gradients_1_local, gradients_2_local),
                    equations_parabolic)
    end
end

function analyze(::typeof(entropy_timederivative), du, u, t,
                 mesh::Union{TreeMesh{2}, StructuredMesh{2}, StructuredMeshView{2},
                             UnstructuredMesh2D, P4estMesh{2}, T8codeMesh{2}},
                 equations, dg::Union{DGSEM, FDSBP}, cache)
    # Calculate ∫(∂S/∂u ⋅ ∂u/∂t)dΩ
    integrate_via_indices(u, mesh, equations, dg, cache,
                          du) do u, i, j, element, equations, dg, du
        u_node = get_node_vars(u, equations, dg, i, j, element)
        du_node = get_node_vars(du, equations, dg, i, j, element)
        return dot(cons2entropy(u_node, equations), du_node)
    end
end

# To calculate the divergence error for the electric field, we need the charge density.
# For the GLMMaxwell equations the charge density is given 
# by the source terms of the Lagrange multiplier equation, so we have to overload at an earlier stage
# to make the source terms available to the analyzer.
function analyze(quantity::Val{:l2_dive}, du, u, t, semi::AbstractSemidiscretization)
    mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
    analyze(quantity::Val{:l2_dive}, du, u, t, semi.source_terms, mesh, equations, solver, cache)
end

function analyze(::Val{:l2_dive}, du, u, t,
                 source_terms, mesh::TreeMesh{2},
                 equations, dg::DGSEM, cache)
    integrate_via_indices(u, mesh, equations, dg, cache, cache,
                          dg.basis.derivative_matrix) do u, i, j, element, equations,
                                                         dg, cache, derivative_matrix
        dive = zero(eltype(u))
        for k in eachnode(dg)
            u_kj = get_node_vars(u, equations, dg, k, j, element)
            u_ik = get_node_vars(u, equations, dg, i, k, element)

            E1_kj, _ = electric_field(u_kj, equations)
            _, E2_ik = electric_field(u_ik, equations)

            dive += (derivative_matrix[i, k] * E1_kj +
                     derivative_matrix[j, k] * E2_ik)

        end
        u_ij = get_node_vars(u, equations, dg, i, j, element)
        x_ij = Trixi.get_node_coords(cache.elements.node_coordinates, equations, dg, i, j,
                                      element)
        dive = cache.elements.inverse_jacobian[element] * dive - scaled_charge_density(u_ij, x_ij, t, source_terms, equations)
        dive^2
    end |> sqrt
end

function analyze(::Val{:l2_dive}, du, u, t,
                 source_terms,
                 mesh::Union{StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2},
                             T8codeMesh{2}},
                 equations, dg::DGSEM, cache)
    @unpack contravariant_vectors = cache.elements
    integrate_via_indices(u, mesh, equations, dg, cache, cache,
                          dg.basis.derivative_matrix) do u, i, j, element, equations,
                                                         dg, cache, derivative_matrix
        dive = zero(eltype(u))
        # Get the contravariant vectors Ja^1 and Ja^2
        Ja11, Ja12 = get_contravariant_vector(1, contravariant_vectors, i, j, element)
        Ja21, Ja22 = get_contravariant_vector(2, contravariant_vectors, i, j, element)
        # Compute the transformed divergence
        for k in eachnode(dg)
            u_kj = get_node_vars(u, equations, dg, k, j, element)
            u_ik = get_node_vars(u, equations, dg, i, k, element)

            E1_kj, E2_kj = electric_field(u_kj, equations)
            E1_ik, E2_ik = electric_field(u_ik, equations)

            dive += (derivative_matrix[i, k] *
                     (Ja11 * E1_kj + Ja12 * E2_kj) +
                     derivative_matrix[j, k] *
                     (Ja21 * E1_ik + Ja22 * E2_ik))
        end
        u_ij = get_node_vars(u, equations, dg, i, j, element)
        x_ij = Trixi.get_node_coords(cache.elements.node_coordinates, equations, dg, i, j,
                                      element)
        dive = cache.elements.inverse_jacobian[i, j, element] * dive - scaled_charge_density(u_ij, x_ij, t, source_terms, equations)
        dive^2
    end |> sqrt
end


function analyze(::Val{:l2_e_normal_jump}, du, u, t,
                 mesh::TreeMesh{2},
                 equations, dg::DGSEM, cache)
    a = integrate_interfaces_via_indices(u, mesh, equations, dg, cache, cache) do u, i, interface, equations, dg, cache                                                
        @unpack u, orientations = cache.interfaces
        normal_jump = zero(eltype(u))
        u_ll, u_rr = get_surface_node_vars(u, equations, dg, i, interface)
        if orientations[interface] == 1
            E1_ll, _ = electric_field(u_ll, equations)
            E1_rr, _ = electric_field(u_rr, equations)
            normal_jump += (E1_ll - E1_rr)^2
        else
            _, E2_ll = electric_field(u_ll, equations)
            _, E2_rr = electric_field(u_rr, equations)
            normal_jump += (E2_ll - E2_rr)^2
        end
    end

    b = integrate_mortars_via_indices(u, mesh, equations, dg, cache, cache) do u, i, mortar, equations, dg, cache                                                        
        @unpack u_upper, u_lower, orientations = cache.mortars
        normal_jump = zero(eltype(u))
        u_upper_ll, u_upper_rr = get_surface_node_vars(u_upper, equations, dg,
                                                        i, mortar)
        u_lower_ll, u_lower_rr = get_surface_node_vars(u_lower, equations, dg,
                                                        i, mortar)
        if orientations[mortar] == 1
            E1u_ll, _ = electric_field(u_upper_ll, equations)
            E1u_rr, _ = electric_field(u_upper_rr, equations)
            E1l_ll, _ = electric_field(u_lower_ll, equations)
            E1l_rr, _ = electric_field(u_lower_rr, equations)
            normal_jump += (E1u_ll - E1u_rr)^2 + (E1l_ll - E1l_rr)^2
        else
            _, E2u_ll = electric_field(u_upper_ll, equations)
            _, E2u_rr = electric_field(u_upper_rr, equations)
            _, E2l_ll = electric_field(u_lower_ll, equations)
            _, E2l_rr = electric_field(u_lower_rr, equations)
            normal_jump += (E2u_ll - E2u_rr)^2 + (E2l_ll - E2l_rr)^2
        end
    end

    return sqrt(a + b)
end

function analyze(::Val{:l2_divb}, du, u, t,
                 mesh::TreeMesh{2},
                 equations, dg::DGSEM, cache)
    integrate_via_indices(u, mesh, equations, dg, cache, cache,
                          dg.basis.derivative_matrix) do u, i, j, element, equations,
                                                         dg, cache, derivative_matrix
        divb = zero(eltype(u))
        for k in eachnode(dg)
            u_kj = get_node_vars(u, equations, dg, k, j, element)
            u_ik = get_node_vars(u, equations, dg, i, k, element)

            B1_kj, _, _ = magnetic_field(u_kj, equations)
            _, B2_ik, _ = magnetic_field(u_ik, equations)

            divb += (derivative_matrix[i, k] * B1_kj +
                     derivative_matrix[j, k] * B2_ik)
        end
        divb *= cache.elements.inverse_jacobian[element]
        divb^2
    end |> sqrt
end

function analyze(::Val{:l2_divb}, du, u, t,
                 mesh::Union{StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2},
                             T8codeMesh{2}},
                 equations, dg::DGSEM, cache)
    @unpack contravariant_vectors = cache.elements
    integrate_via_indices(u, mesh, equations, dg, cache, cache,
                          dg.basis.derivative_matrix) do u, i, j, element, equations,
                                                         dg, cache, derivative_matrix
        divb = zero(eltype(u))
        # Get the contravariant vectors Ja^1 and Ja^2
        Ja11, Ja12 = get_contravariant_vector(1, contravariant_vectors, i, j, element)
        Ja21, Ja22 = get_contravariant_vector(2, contravariant_vectors, i, j, element)
        # Compute the transformed divergence
        for k in eachnode(dg)
            u_kj = get_node_vars(u, equations, dg, k, j, element)
            u_ik = get_node_vars(u, equations, dg, i, k, element)

            B1_kj, B2_kj, _ = magnetic_field(u_kj, equations)
            B1_ik, B2_ik, _ = magnetic_field(u_ik, equations)

            divb += (derivative_matrix[i, k] *
                     (Ja11 * B1_kj + Ja12 * B2_kj) +
                     derivative_matrix[j, k] *
                     (Ja21 * B1_ik + Ja22 * B2_ik))
        end
        divb *= cache.elements.inverse_jacobian[i, j, element]
        divb^2
    end |> sqrt
end

function analyze(::Val{:linf_divb}, du, u, t,
                 mesh::TreeMesh{2},
                 equations, dg::DGSEM, cache)
    @unpack derivative_matrix, weights = dg.basis

    # integrate over all elements to get the divergence-free condition errors
    linf_divb = zero(eltype(u))
    @batch reduction=(max, linf_divb) for element in eachelement(dg, cache)
        for j in eachnode(dg), i in eachnode(dg)
            divb = zero(eltype(u))
            for k in eachnode(dg)
                u_kj = get_node_vars(u, equations, dg, k, j, element)
                u_ik = get_node_vars(u, equations, dg, i, k, element)

                B1_kj, _, _ = magnetic_field(u_kj, equations)
                _, B2_ik, _ = magnetic_field(u_ik, equations)

                divb += (derivative_matrix[i, k] * B1_kj +
                         derivative_matrix[j, k] * B2_ik)
            end
            divb *= cache.elements.inverse_jacobian[element]
            linf_divb = max(linf_divb, abs(divb))
        end
    end

    return linf_divb
end

function analyze(::Val{:linf_divb}, du, u, t,
                 mesh::Union{StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2},
                             T8codeMesh{2}},
                 equations, dg::DGSEM, cache)
    @unpack derivative_matrix, weights = dg.basis
    @unpack contravariant_vectors = cache.elements

    # integrate over all elements to get the divergence-free condition errors
    linf_divb = zero(eltype(u))
    @batch reduction=(max, linf_divb) for element in eachelement(dg, cache)
        for j in eachnode(dg), i in eachnode(dg)
            divb = zero(eltype(u))
            # Get the contravariant vectors Ja^1 and Ja^2
            Ja11, Ja12 = get_contravariant_vector(1, contravariant_vectors,
                                                  i, j, element)
            Ja21, Ja22 = get_contravariant_vector(2, contravariant_vectors,
                                                  i, j, element)
            # Compute the transformed divergence
            for k in eachnode(dg)
                u_kj = get_node_vars(u, equations, dg, k, j, element)
                u_ik = get_node_vars(u, equations, dg, i, k, element)

                B1_kj, B2_kj, _ = magnetic_field(u_kj, equations)
                B1_ik, B2_ik, _ = magnetic_field(u_ik, equations)

                divb += (derivative_matrix[i, k] *
                         (Ja11 * B1_kj + Ja12 * B2_kj) +
                         derivative_matrix[j, k] *
                         (Ja21 * B1_ik + Ja22 * B2_ik))
            end
            divb *= cache.elements.inverse_jacobian[i, j, element]
            linf_divb = max(linf_divb, abs(divb))
        end
    end
    if mpi_isparallel()
        # Base.max instead of max needed, see comment in src/auxiliary/math.jl
        linf_divb = MPI.Allreduce!(Ref(linf_divb), Base.max, mpi_comm())[]
    end

    return linf_divb
end
end # @muladd
