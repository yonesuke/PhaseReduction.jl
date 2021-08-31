mutable struct HodgkinHuxley <: SpecialDiffSys
    C::Float64
    G_Na::Float64
    G_K::Float64
    G_L::Float64
    E_Na::Float64
    E_K::Float64
    E_L::Float64
    I_bias::Float64
    noise::AbstractArray
end

name(::HodgkinHuxley) = "Hodgkin-Huxley equation"
dimension(::HodgkinHuxley) = 4

HodgkinHuxley(; C=1.0, G_Na=120.0, G_K=36, G_L=0.3, E_Na=50.0, E_K=-77.0, E_L=-54.4, I_bias=30, noise=zeros(4)) = HodgkinHuxley(C, G_Na, G_K, G_L, E_Na, E_K, E_L, I_bias, noise)

function vec_field(hh::HodgkinHuxley, x::AbstractArray)
    V, m, h, n = x
    C = hh.C
    G_Na, G_K, G_L = hh.G_Na, hh.G_K, hh.G_L
    E_Na, E_K, E_L = hh.E_Na, hh.E_K, hh.E_L
    I_bias = hh.I_bias
    _dV = G_Na*m^3*h*(E_Na-V) + G_K*n^4*(E_K-V) + G_L*(E_L-V) + I_bias
    dV = _dV/C
    αₘ = 0.1(V+40)/(1-exp((-V-40)/10))
    βₘ = 4exp((-V-65)/18)
    dm = αₘ*(1-m) - βₘ*m
    αₕ = 0.07exp((-V-65)/20)
    βₕ = 1/(1+exp((-V-35)/10))
    dh = αₕ*(1-h) - βₕ*h
    αₙ = 0.01(V+55)/(1-exp((-V-55)/10))
    βₙ = 0.125exp((-V-65)/80)
    dn = αₙ*(1-n) - βₙ*n
    return [dV, dm, dh, dn]
end

function jacobian(hh::HodgkinHuxley, x::AbstractArray)
    V, m, h, n = x
    C = hh.C
    G_Na, G_K, G_L = hh.G_Na, hh.G_K, hh.G_L
    E_Na, E_K, E_L = hh.E_Na, hh.E_K, hh.E_L
    I_bias = hh.I_bias
end