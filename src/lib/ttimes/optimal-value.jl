using IntervalArithmetic
using JuMP
import HiGHS

include("../crisp-pcm.jl")
include("../nearly-equal.jl")

LPResult_Crisp = @NamedTuple{
    # 区間重みベクトル
    wᴸ_center_1::Vector{T}, wᵁ_center_1::Vector{T},
    W_center_1::Vector{Interval{T}}, # ([Wᵢ_center_1ᴸ, Wᵢ_center_1ᵁ])
    optimalValue_center_1::T
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
        # wᵢ_center_1ᴸ ≥ ε, wᵢ_center_1ᵁ ≥ ε
        @variable(model, wᴸ_center_1[i=1:n] ≥ ε); @variable(model, wᵁ_center_1[i=1:n] ≥ ε)

        # 上三角成分に対応する i, j
        for i = 1:n-1
            wᵢ_center_1ᴸ = wᴸ_center_1[i]; wᵢ_center_1ᵁ = wᵁ_center_1[i]

            for j = i+1:n
                aᵢⱼ = A[i,j]
                wⱼ_center_1ᴸ = wᴸ_center_1[j]; wⱼ_center_1ᵁ = wᵁ_center_1[j]

                @constraint(model, wᵢ_center_1ᴸ ≤ aᵢⱼ * wⱼ_center_1ᵁ)
                @constraint(model, aᵢⱼ * wⱼ_center_1ᴸ ≤ wᵢ_center_1ᵁ)
            end
        end

        for i = 1:n
            wᵢ_center_1ᴸ = wᴸ_center_1[i]; wᵢ_center_1ᵁ = wᵁ_center_1[i]
            
            # 正規性条件
            ∑wⱼ_center_1ᴸ = sum(map(j -> wᴸ_center_1[j], filter(j -> i != j, 1:n)))
            @constraint(model, ∑wⱼ_center_1ᴸ + wᵢ_center_1ᵁ ≤ 1)
            ∑wⱼ_center_1ᵁ = sum(map(j -> wᵁ_center_1[j], filter(j -> i != j, 1:n)))
            @constraint(model, ∑wⱼ_center_1ᵁ + wᵢ_center_1ᴸ ≥ 1)

            @constraint(model, wᵢ_center_1ᵁ ≥ wᵢ_center_1ᴸ)
        end

        @constraint(model, sum(wᵁ_center_1) + sum(wᴸ_center_1) == 2)

        # 目的関数 ∑(wᵢ_center_1ᵁ - wᵢ_center_1ᴸ)
        @objective(model, Min, sum(wᵁ_center_1) - sum(wᴸ_center_1))

        optimize!(model)

        optimalValue_center_1 = sum(value.(wᵁ_center_1)) - sum(value.(wᴸ_center_1))

        wᴸ_center_1_value = value.(wᴸ_center_1)
        wᵁ_center_1_value = value.(wᵁ_center_1)

        # precision error 対応
        for i = 1:n
            if wᴸ_center_1_value[i] > wᵁ_center_1_value[i]
                wᴸ_center_1_value[i] = wᵁ_center_1_value[i]
            end
        end
        W_center_1_value = map(i -> (wᴸ_center_1_value[i])..(wᵁ_center_1_value[i]), 1:n)

        return (
            wᴸ_center_1=wᴸ_center_1_value, wᵁ_center_1=wᵁ_center_1_value,
            W_center_1=W_center_1_value,
            optimalValue_center_1=optimalValue_center_1
        )
    finally
        # エラー終了時にも変数などを消去する
        empty!(model)
    end
end
