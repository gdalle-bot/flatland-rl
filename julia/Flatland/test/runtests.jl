using Aqua: Aqua
using ExplicitImports: ExplicitImports
using Flatland
using JET: JET
using Test

@testset verbose = true "Flatland.jl" begin
    @testset verbose = true "Code quality" begin
        @testset "Aqua" begin
            Aqua.test_all(Flatland; undocumented_names = VERSION >= v"1.11")
        end
        @testset "ExplicitImports" begin
            ExplicitImports.test_explicit_imports(Flatland)
        end
        @testset "JET" begin
            JET.test_package(Flatland; target_modules = (Flatland,))
        end
    end
end
