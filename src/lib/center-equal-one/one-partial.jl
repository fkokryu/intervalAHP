using IntervalArithmetic
using JuMP
import HiGHS

include("../crisp-pcm.jl")
include("../nearly-equal.jl")
include("../ttimes/optimal-value.jl")

LPResult_one_PartialIncorporation = @NamedTuple{
    # 区間重みベクトル
    wᴸ_partial_center_1::Vector{T}, wᵁ_partial_center_1::Vector{T},
    W_partial_center_1::Vector{Interval{T}}, # ([wᵢᴸ_partial_center_1, wᵢᵁ_partial_center_1])
    ŵᴸ_partial_center_1::Matrix{T}, ŵᵁ_partial_center_1::Matrix{T},
    Ŵ_partial_center_1::Vector{Vector{Interval{T}}}, # ([ŵᵢᴸ, ŵᵢᵁ])
    w_partial_center_1::Vector{Vector{T}},
    optimalValue_partial_center_1::T
    } where {T <: Real}

function solveonePartialIncorporationLP(
        matrices::Vector{Matrix{T}}
        )::Union{LPResult_one_PartialIncorporation{T}, Nothing} where {T <: Real}
    ε = 1e-8 # << 1

    if isempty(matrices)
        throw(ArgumentError("Argument matrices is empty"))
    end

    if !all(Aₖ -> isCrispPCM(Aₖ), matrices)
        throw(ArgumentError("Aₖ is not a crisp PCM"))
    end

    l = length(matrices) # 人数
    m, n = size(matrices[1])

    if !all(Aₖ -> size(Aₖ) == (n, n), matrices)
        throw(ArgumentError("Some matrices have different size"))
    end

    ḋ_partial_center_1 = map(Aₖ -> solveCrispAHPLP(Aₖ).optimalValue_center_1, matrices)

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    try
        # wᵢᴸ_partial_center_1 ≥ ε, wᵢᵁ_partial_center_1 ≥ ε
        @variable(model, wᴸ_partial_center_1[i=1:n] ≥ ε)
        @variable(model, wᵁ_partial_center_1[i=1:n] ≥ ε)
        # ŵₖᵢᴸ_partial_center_1 ≥ ε, ŵₖᵢᵁ_partial_center_1 ≥ ε
        @variable(model, ŵᴸ_partial_center_1[k=1:l,i=1:n] ≥ ε)
        @variable(model, ŵᵁ_partial_center_1[k=1:l,i=1:n] ≥ ε)
        # wₖ_partial_center_1ᵢ ≥ ε
        @variable(model, w_partial_center_1[k=1:l,i=1:n] ≥ ε)

        for k = 1:l
            ŵₖᴸ_partial_center_1 = ŵᴸ_partial_center_1[k,:]; ŵₖᵁ_partial_center_1 = ŵᵁ_partial_center_1[k,:]
            wₖ_partial_center_1 = w_partial_center_1[k,:]

            Aₖ = matrices[k]

            # ∑(ŵₖᵢᵁ_partial_center_1 - ŵₖᵢᴸ_partial_center_1) ≤ ḋ_partial_center_1ₖ
            @constraint(model, sum(ŵₖᵁ_partial_center_1) - sum(ŵₖᴸ_partial_center_1) ≤ (ḋ_partial_center_1[k] + ε))

            for i = 1:n-1
                ŵₖᵢᴸ_partial_center_1 = ŵₖᴸ_partial_center_1[i]; ŵₖᵢᵁ_partial_center_1 = ŵₖᵁ_partial_center_1[i]

                for j = i+1:n
                    aₖᵢⱼ = Aₖ[i,j]
                    ŵₖⱼᴸ_partial_center_1 = ŵₖᴸ_partial_center_1[j]; ŵₖⱼᵁ_partial_center_1 = ŵₖᵁ_partial_center_1[j]

                    @constraint(model, ŵₖᵢᴸ_partial_center_1 ≤ aₖᵢⱼ * ŵₖⱼᵁ_partial_center_1)
                    @constraint(model, aₖᵢⱼ * ŵₖⱼᴸ_partial_center_1 ≤ ŵₖᵢᵁ_partial_center_1)
                end
            end

            # 正規性条件
            @constraint(model, sum(wₖ_partial_center_1) == 1)

            for i = 1:n
                ŵₖᵢᴸ_partial_center_1 = ŵₖᴸ_partial_center_1[i]; ŵₖᵢᵁ_partial_center_1 = ŵₖᵁ_partial_center_1[i]
                wᵢᴸ_partial_center_1 = wᴸ_partial_center_1[i]; wᵢᵁ_partial_center_1 = wᵁ_partial_center_1[i]
                wₖ_partial_center_1ᵢ = wₖ_partial_center_1[i]

                # 正規性条件
                ∑ŵₖⱼᴸ_partial_center_1 = sum(map(j -> ŵₖᴸ_partial_center_1[j], filter(j -> i != j, 1:n)))
                @constraint(model, ∑ŵₖⱼᴸ_partial_center_1 + ŵₖᵢᵁ_partial_center_1 ≤ 1)
                ∑ŵₖⱼᵁ_partial_center_1 = sum(map(j -> ŵₖᵁ_partial_center_1[j], filter(j -> i != j, 1:n)))
                @constraint(model, ∑ŵₖⱼᵁ_partial_center_1 + ŵₖᵢᴸ_partial_center_1 ≥ 1)

                @constraint(model, wₖ_partial_center_1ᵢ ≥ wᵢᴸ_partial_center_1)
                @constraint(model, ŵₖᵢᵁ_partial_center_1 ≥ wₖ_partial_center_1ᵢ)

                @constraint(model, wₖ_partial_center_1ᵢ ≥ ŵₖᵢᴸ_partial_center_1)
                @constraint(model, wᵢᵁ_partial_center_1 ≥ wₖ_partial_center_1ᵢ)
            end

            @constraint(model, sum(ŵₖᴸ_partial_center_1) + sum(ŵₖᵁ_partial_center_1) == 2)
        end

        @constraint(model, sum(wᵁ_partial_center_1) + sum(wᴸ_partial_center_1) == 2)

        # 目的関数 ∑(wᵢᵁ_partial_center_1 - wᵢᴸ_partial_center_1)
        @objective(model, Min, sum(wᵁ_partial_center_1) - sum(wᴸ_partial_center_1))

        optimize!(model)

        if termination_status(model) == MOI.OPTIMAL
            # 解が見つかった場合の処理
            optimalValue_partial_center_1 = sum(value.(wᵁ_partial_center_1)) - sum(value.(wᴸ_partial_center_1))

            wᴸ_partial_center_1_value = value.(wᴸ_partial_center_1); wᵁ_partial_center_1_value = value.(wᵁ_partial_center_1)
            # precision error 対応
            for i = 1:n
                if wᴸ_partial_center_1_value[i] > wᵁ_partial_center_1_value[i]
                    wᴸ_partial_center_1_value[i] = wᵁ_partial_center_1_value[i]
                end
            end
            W_partial_center_1_value = map(i -> (wᴸ_partial_center_1_value[i])..(wᵁ_partial_center_1_value[i]), 1:n)

            ŵᴸ_partial_center_1_value = value.(ŵᴸ_partial_center_1); ŵᵁ_partial_center_1_value = value.(ŵᵁ_partial_center_1)
            # precision error 対応
            for k = 1:l, i = 1:n
                if ŵᴸ_partial_center_1_value[k,i] > ŵᵁ_partial_center_1_value[k,i]
                    ŵᴸ_partial_center_1_value[k,i] = ŵᵁ_partial_center_1_value[k,i]
                end
            end
            Ŵ_partial_center_1_value = map(
                k -> map(i -> (ŵᴸ_partial_center_1_value[k,i])..(ŵᵁ_partial_center_1_value[k,i]), 1:n),
                1:l)

                w_partial_center_1_value = map(k -> value.(w_partial_center_1[k,:]), 1:l)

            return (
                wᴸ_partial_center_1=wᴸ_partial_center_1_value, wᵁ_partial_center_1=wᵁ_partial_center_1_value,
                W_partial_center_1=W_partial_center_1_value,
                ŵᴸ_partial_center_1=ŵᴸ_partial_center_1_value, ŵᵁ_partial_center_1=ŵᵁ_partial_center_1_value,
                Ŵ_partial_center_1=Ŵ_partial_center_1_value,
                w_partial_center_1=w_partial_center_1_value,
                optimalValue_partial_center_1=optimalValue_partial_center_1
            )
        else
            # 解が見つからなかった場合の処理
            println("The PartialIncorporation optimization problem had no optimal solution.")
            return nothing  # 解が見つからなかったことを示すためにnothingを返す
        end

    catch e
        # エラーが発生した場合の処理
        println("An error occurred during optimization: ", e)
        return nothing  # エラーが発生したことを示すためにnothingを返す
    end
end
