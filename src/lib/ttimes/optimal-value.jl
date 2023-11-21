using IntervalArithmetic
using JuMP
import HiGHS

include("../crisp-pcm.jl")
include("../nearly-equal.jl")

LPResult_Crisp = @NamedTuple{
    # 区間重みベクトル
    vᴸ::Vector{T}, vᵁ::Vector{T},
    V::Vector{Interval{T}}, # ([Vᵢᴸ, Vᵢᵁ])
    optimalValue::T
    } where {T <: Real}

function solveCrispAHPLP(A::Matrix{T})::LPResult_Crisp{T} where {T <: Real}
    ε = 1e-8 # << 1

    if !isCrispPCM(A)
        throw(ArgumentError("A is not a crisp PCM"))
    end

    m, n = size(A)
    model = Model(HiGHS.Optimizer)
    set_silent(model)

    try
        # vᵢᴸ ≥ ε, vᵢᵁ ≥ ε
        @variable(model, vᴸ[i=1:n] ≥ ε); @variable(model, vᵁ[i=1:n] ≥ ε)

        # 上三角成分に対応する i, j
        for i = 1:n-1
            vᵢᴸ = vᴸ[i]; vᵢᵁ = vᵁ[i]

            for j = i+1:n
                aᵢⱼ = A[i,j]
                vⱼᴸ = vᴸ[j]; vⱼᵁ = vᵁ[j]

                @constraint(model, vᵢᴸ ≤ aᵢⱼ * vⱼᵁ)
                @constraint(model, aᵢⱼ * vⱼᴸ ≤ vᵢᵁ)
            end
        end

        for i = 1:n
            vᵢᴸ = vᴸ[i]; vᵢᵁ = vᵁ[i]
            
            # 正規性条件
            ∑vⱼᴸ = sum(map(j -> vᴸ[j], filter(j -> i != j, 1:n)))
            @constraint(model, ∑vⱼᴸ + vᵢᵁ ≤ 1)
            ∑vⱼᵁ = sum(map(j -> vᵁ[j], filter(j -> i != j, 1:n)))
            @constraint(model, ∑vⱼᵁ + vᵢᴸ ≥ 1)

            @constraint(model, vᵢᵁ ≥ vᵢᴸ)
        end

        @constraint(model, sum(vᵁ) + sum(vᴸ) == 2)

        # 目的関数 ∑(vᵢᵁ - vᵢᴸ)
        @objective(model, Min, sum(vᵁ) - sum(vᴸ))

        optimize!(model)

        optimalValue = sum(value.(vᵁ)) - sum(value.(vᴸ))

        vᴸ_value = value.(vᴸ)
        vᵁ_value = value.(vᵁ)

        # precision error 対応
        for i = 1:n
            if vᴸ_value[i] > vᵁ_value[i]
                vᴸ_value[i] = vᵁ_value[i]
            end
        end
        V_value = map(i -> (vᴸ_value[i])..(vᵁ_value[i]), 1:n)

        return (
            vᴸ=vᴸ_value, vᵁ=vᵁ_value,
            V=V_value,
            optimalValue=optimalValue
        )
    finally
        # エラー終了時にも変数などを消去する
        empty!(model)
    end
end
