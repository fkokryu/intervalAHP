using IntervalArithmetic
using JuMP
import HiGHS

include("../crisp-pcm.jl")
include("../nearly-equal.jl")
include("../ttimes/optimal-value.jl")

LPResult_one_CommonGround = @NamedTuple{
    # 区間重みベクトル
    wᴸ_common_center_1::Vector{T}, wᵁ_common_center_1::Vector{T},
    W_common_center_1::Vector{Interval{T}}, # ([wᵢᴸ_common_center_1, wᵢᵁ_common_center_1])
    ŵᴸ_common_center_1::Matrix{T}, ŵᵁ_common_center_1::Matrix{T},
    Ŵ_common_center_1::Vector{Vector{Interval{T}}}, # ([ŵᵢᴸ, ŵᵢᵁ])
    optimalValue_common_center_1::T
    } where {T <: Real}

function solveoneCommonGroundLP(matrices::Vector{Matrix{T}})::Union{LPResult_one_CommonGround{T}, Nothing} where {T <: Real}
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

    ḋ_common_center_1 = map(Aₖ -> solveCrispAHPLP(Aₖ).optimalValue_center_1, matrices)

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    try
        # wᵢᴸ_common_center_1 ≥ ε, wᵢᵁ_common_center_1 ≥ ε
        @variable(model, wᴸ_common_center_1[i=1:n] ≥ ε); @variable(model, wᵁ_common_center_1[i=1:n] ≥ ε)
        # ŵₖᵢᴸ_common_center_1 ≥ ε, ŵₖᵢᵁ_common_center_1 ≥ ε
        @variable(model, ŵᴸ_common_center_1[k=1:l,i=1:n] ≥ ε); @variable(model, ŵᵁ_common_center_1[k=1:l,i=1:n] ≥ ε)

        for k = 1:l
            ŵₖᴸ_common_center_1 = ŵᴸ_common_center_1[k,:]; ŵₖᵁ_common_center_1 = ŵᵁ_common_center_1[k,:]

            Aₖ = matrices[k]

            # ∑(ŵₖᵢᵁ_common_center_1 - ŵₖᵢᴸ_common_center_1) ≤ ḋ_common_center_1ₖ
            @constraint(model, sum(ŵₖᵁ_common_center_1) - sum(ŵₖᴸ_common_center_1) ≤ ḋ_common_center_1[k])

            for i = 1:n-1
                ŵₖᵢᴸ_common_center_1 = ŵₖᴸ_common_center_1[i]; ŵₖᵢᵁ_common_center_1 = ŵₖᵁ_common_center_1[i]
    
                for j = i+1:n
                    aₖᵢⱼ = Aₖ[i,j]
                    ŵₖⱼᴸ_common_center_1 = ŵₖᴸ_common_center_1[j]; ŵₖⱼᵁ_common_center_1 = ŵₖᵁ_common_center_1[j]
                    
                    @constraint(model, ŵₖᵢᴸ_common_center_1 ≤ aₖᵢⱼ * ŵₖⱼᵁ_common_center_1)
                    @constraint(model, aₖᵢⱼ * ŵₖⱼᴸ_common_center_1 ≤ ŵₖᵢᵁ_common_center_1)
                end
            end

            for i = 1:n
                ŵₖᵢᴸ_common_center_1 = ŵₖᴸ_common_center_1[i]; ŵₖᵢᵁ_common_center_1 = ŵₖᵁ_common_center_1[i]

                # 正規性条件
                ∑ŵₖⱼᴸ_common_center_1 = sum(map(j -> ŵₖᴸ_common_center_1[j], filter(j -> i != j, 1:n)))
                @constraint(model, ∑ŵₖⱼᴸ_common_center_1 + ŵₖᵢᵁ_common_center_1 ≤ 1)
                ∑ŵₖⱼᵁ_common_center_1 = sum(map(j -> ŵₖᵁ_common_center_1[j], filter(j -> i != j, 1:n)))
                @constraint(model, ∑ŵₖⱼᵁ_common_center_1 + ŵₖᵢᴸ_common_center_1 ≥ 1)

                wᵢᴸ_common_center_1 = wᴸ_common_center_1[i]; wᵢᵁ_common_center_1 = wᵁ_common_center_1[i]
                @constraint(model, wᵢᴸ_common_center_1 ≥ ŵₖᵢᴸ_common_center_1)
                @constraint(model, wᵢᵁ_common_center_1 ≥ wᵢᴸ_common_center_1)
                @constraint(model, ŵₖᵢᵁ_common_center_1 ≥ wᵢᵁ_common_center_1)
            end

            @constraint(model, sum(ŵₖᴸ_common_center_1) + sum(ŵₖᵁ_common_center_1) == 2)
        end

        for i = 1:n
            wᵢᴸ_common_center_1 = wᴸ_common_center_1[i]; wᵢᵁ_common_center_1 = wᵁ_common_center_1[i] 
            # kなしの正規性条件
            ∑wⱼᴸ_common_center_1 = sum(map(j -> wᴸ_common_center_1[j], filter(j -> i != j, 1:n)))
            @constraint(model, ∑wⱼᴸ_common_center_1 + wᵢᵁ_common_center_1 ≤ 1)
            ∑wⱼᵁ_common_center_1 = sum(map(j -> wᵁ_common_center_1[j], filter(j -> i != j, 1:n)))
            @constraint(model, ∑wⱼᵁ_common_center_1 + wᵢᴸ_common_center_1 ≥ 1)
        end

        @constraint(model, sum(wᵁ_common_center_1) + sum(wᴸ_common_center_1) == 2)

        # 目的関数 ∑(wᵢᵁ_common_center_1 - wᵢᴸ_common_center_1)
        @objective(model, Max, sum(wᵁ_common_center_1) - sum(wᴸ_common_center_1))

        optimize!(model)

        if termination_status(model) == MOI.OPTIMAL
            # 解が見つかった場合の処理
            optimalValue_common_center_1 = sum(value.(wᵁ_common_center_1)) - sum(value.(wᴸ_common_center_1))

            wᴸ_common_center_1_value = value.(wᴸ_common_center_1)
            wᵁ_common_center_1_value = value.(wᵁ_common_center_1)
            # precision error 対応
            for i = 1:n
                if wᴸ_common_center_1_value[i] > wᵁ_common_center_1_value[i]
                    wᴸ_common_center_1_value[i] = wᵁ_common_center_1_value[i]
                end
            end
            W_common_center_1_value = map(i -> (wᴸ_common_center_1_value[i])..(wᵁ_common_center_1_value[i]), 1:n)
            
            ŵᴸ_common_center_1_value = value.(ŵᴸ_common_center_1)
            ŵᵁ_common_center_1_value = value.(ŵᵁ_common_center_1)
            # precision error 対応
            for k = 1:l, i = 1:n
                if ŵᴸ_common_center_1_value[k,i] > ŵᵁ_common_center_1_value[k,i]
                    ŵᴸ_common_center_1_value[k,i] = ŵᵁ_common_center_1_value[k,i]
                end
            end
            Ŵ_common_center_1_value = map(
                k -> map(i -> (ŵᴸ_common_center_1_value[k,i])..(ŵᵁ_common_center_1_value[k,i]), 1:n),
                1:l)

            return (
                wᴸ_common_center_1=wᴸ_common_center_1_value, wᵁ_common_center_1=wᵁ_common_center_1_value,
                W_common_center_1=W_common_center_1_value,
                ŵᴸ_common_center_1=ŵᴸ_common_center_1_value, ŵᵁ_common_center_1=ŵᵁ_common_center_1_value,
                Ŵ_common_center_1=Ŵ_common_center_1_value,
                optimalValue_common_center_1=optimalValue_common_center_1
            )
        else
            # 解が見つからなかった場合の処理
            println("The CommonGround optimization problem had no optimal solution.")
            return nothing  # 解が見つからなかったことを示すためにnothingを返す
        end

    catch e
        # エラーが発生した場合の処理
        println("An error occurred during optimization: ", e)
        return nothing  # エラーが発生したことを示すためにnothingを返す
    end
end
