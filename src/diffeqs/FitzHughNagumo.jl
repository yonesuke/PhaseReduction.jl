mutable struct FitzHughNagumo <: SpecialDiffSys
    δ::Float64
    a::Float64
    b::Float64
    I::Float64
    noise::AbstractArray
end

name(::FitzHughNagumo) = "Fitz-Hugh Nagumo equation"
dimension(::FitzHughNagumo) = 2

FitzHughNagumo(; δ=0.1, a=1.0, b=0.8, I=0.8, noise=zeros(2)) = FitzHughNagumo(δ, a, b, I, noise)

function vec_field(fhn::FitzHughNagumo, x::AbstractArray)
    u, v = x
    δ = fhn.δ
    a = fhn.a
    b = fhn.b
    I = fhn.I
    du = δ*(v+a-b*u)
    dv = v-v^3/3.0-u+I
    return [du, dv]
end

function jacobian(fhn::FitzHughNagumo, x::AbstractArray)
    _, v = x
    δ = fhn.δ
    b = fhn.b
    J = [-δ*b δ; -1 1-v*v]
    return J
end
