module PhaseReduction

export AbstractDiffSys, SpecialDiffSys, DiffSys, CoupledDiffSys, integrate,
    PeriodicOrbit, PhaseSensitivity, PhaseCoupling,
    vec_field, jacobian,
    vanderPol, FitzHughNagumo, HodgkinHuxley, FastSpikingNeuron

using LinearAlgebra

include("utils.jl")
include("reduction.jl")
include("diffeqs/vanderPol.jl")
include("diffeqs/FitzHughNagumo.jl")
include("diffeqs/HodgkinHuxley.jl")
include("diffeqs/FastSpikingNeuron.jl")

end