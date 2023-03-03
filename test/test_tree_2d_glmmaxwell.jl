module TestExamples2DGLMMaxwell

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "tree_2d_dgsem")

@testset "GLM Maxwell" begin
  @trixi_testset "elixir_advection_basic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
      # Expected errors are exactly the same as in the parallel test!
      l2   = [8.311947673061856e-6],
      linf = [6.627000273229378e-5],
      # Let the small basic test run to the end
      coverage_override = (maxiters=10^5,))
  end

end # module
