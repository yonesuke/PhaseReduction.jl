function PeriodicOrbit(sys::DiffSys; ps=(2,0.0), eps=10^(-10), transient=10^4, steps = 2^12)
    n_dim = sys.n_dim
    x0 = rand(n_dim)
    dt = 0.01
    next_step(x, dt) = RungeKutta(sys.func, dt, x)
    for _ in 1:dt:transient
        x0 = next_step(x0, dt)
    end
    x_prev, x_now = x0, next_step(x0, dt)
    idx, section = ps
    while true
        if (x_prev[idx]>section) & (x_now[idx]<section)
            if dt < eps
                break
            else
                dt *= 0.5
                x_now = next_step(x_prev, dt)
                continue
            end
        end
        x_prev = x_now
        x_now = next_step(x_prev, dt)
    end
    x_dest = x_now
    @info "Poincare section for the system crossing x[$idx]=$section from positive to negative\nperiodic section" x_dest
    t, dt = 0.0, 0.01
    x_prev = x_now
    x_now = next_step(x_prev, dt)
    while true
        if (x_prev[idx]>section) & (x_now[idx]<section)
            if dt < eps
                t += dt
                break
            else
                dt *= 0.5
                x_now = next_step(x_prev, dt)
                continue
            end
        end
        x_prev = x_now
        x_now = next_step(x_prev, dt)
        t += dt
    end
    period = t
    dt = period / steps
    ts = [0:dt:period;]
    x = x_dest
    orbits = zeros(steps+1, n_dim)
    orbits[1,:] = x
    for i in 1:steps
        x = next_step(x, dt)
        orbits[i+1,:] = x
    end
    distance = norm(orbits[begin,:]-orbits[end,:])
    @info "Periodicity checked: distance between start and end of orbit = $distance"
    return period, ts, orbits
end

PeriodicOrbit(sys::SpecialDiffSys; ps=(2,0.0), eps=10^(-10), transient=10^4, steps = 2^12) = PeriodicOrbit(DiffSys(sys), ps=ps, eps=eps, transient=transient, steps = steps)

function PhaseSensitivity(sys::SpecialDiffSys, period::Float64, periodic_orbits::AbstractArray; eps=10^(-12))
    func(x) = vec_field(sys, x)
    jac(x) = jacobian(sys, x)
    ω = 2π/period
    steps, n_dim = size(periodic_orbits)
    dt = period/(steps-1)
    periodic_orbits_reverse = reverse(periodic_orbits, dims=1)
    adjoint_func(z, x) = -jac(x)'*z
    Z_now = zeros(steps, n_dim)
    z = randn(n_dim)
    # contour integral
    for i in 1:steps
        Z_now[i,:] = z
        z -= dt*adjoint_func(z, periodic_orbits_reverse[i,:])
    end
    # normalization
    for i in 1:steps
        @views coeff = ω / dot(Z_now[i,:], func(periodic_orbits_reverse[i, :]))
        Z_now[i,:] *= coeff
    end
    Z_prev = copy(Z_now)
    while true
        Z_now[1,:] = Z_prev[end,:]
        # contour integral
        for i in 1:steps
            Z_now[i,:] = z
            z -= dt*adjoint_func(z, periodic_orbits_reverse[i,:])
        end
        # normalization
        for i in 1:steps
            @views coeff = ω / dot(Z_now[i,:], func(periodic_orbits_reverse[i, :]))
            Z_now[i,:] *= coeff
        end
        # checking convergence
        L∞ = maximum(Z_now-Z_prev)
        if L∞ < eps
            @info "L^∞  norm between phase sensitivity function = $L∞ \n well converged"
            break
        else
            # @info "L^∞ norm between phase sensitivity function = $L∞ \n one more contour integral"
        end
        Z_prev = copy(Z_now)
    end
    return reverse(Z_now, dims=1)
end

function PhaseSensitivity(sys::SpecialDiffSys; steps=2^12)
    period, _, periodic_orbits = PeriodicOrbit(sys, steps=steps)
    @info "Period of periodic orbit" period
    return PhaseSensitivity(sys, period, periodic_orbits)
end

function PhaseCoupling(phasesensitivity::AbstractArray, coupling::Function, periodic_orbit1::AbstractArray, periodic_orbit2::AbstractArray)
    steps = size(phasesensitivity, 1)
    step1, step2 = size(periodic_orbit1, 1), size(periodic_orbit1, 1)
    @assert step1==steps "Time step of phasesensitivity function ($steps) differs from first periodic orbit ($step1)"
    @assert step2==steps "Time step of phasesensitivity function ($steps) differs from first periodic orbit ($step2)"
    dθ = 2π/(steps-1)
    shift_phase(orbits, j) = begin
        tmp = circshift(orbits[begin:end-1,:], -j)
        return vcat(tmp, tmp[1,:]')
    end
    Γ_arr = zeros(steps)
    trapezoidal_weight = ones(steps); trapezoidal_weight[[begin,end]] .= 0.5
    for i in 1:steps
        shifted_periodic_orbit2 = shift_phase(periodic_orbit2, i)
        @views Gjks = [coupling(periodic_orbit1[l,:], shifted_periodic_orbit2[l,:]) for l in 1:steps]
        @views integrands = [dot(phasesensitivity[l,:], Gjks[l]) for l in 1:steps]
        Γ_arr[i] = dot(trapezoidal_weight, integrands)*dθ/2π
    end
    return Γ_arr
end

function PhaseCoupling(sys::CoupledDiffSys)
    periodic_orbit_data = [PeriodicOrbit(s) for s in sys.SelfDiffSyss]
    periodic_orbits = [d[3] for d in periodic_orbit_data]
    phasesensitivitys = [PhaseSensitivity(sys.OriginalSpecialDiffSyss[i], periodic_orbit_data[i][1], periodic_orbit_data[i][3]) for i in 1:sys.n_body]
    phasecoupling_dict = Dict()
    for (key, coupling) in sys.CouplingFuncDict
        sender, reciever = key
        Γ_arr = PhaseCoupling(phasesensitivitys[sender], coupling, periodic_orbits[sender], periodic_orbits[reciever])
        phasecoupling_dict[key] = Γ_arr
    end
    return phasecoupling_dict
end