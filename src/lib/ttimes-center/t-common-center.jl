using IntervalArithmetic
using JuMP
import HiGHS

include("../crisp-pcm.jl")
include("../nearly-equal.jl")
include("../ttimes/optimal-value.jl")

LPResult_t_2_CommonGround = @NamedTuple{
    # 区間重みベクトル
    wᴸ_tcommon_center_1::Vector{T}, wᵁ_tcommon_center_1::Vector{T},
    W_tcommon_center_1::Vector{Interval{T}}, # ([wᵢᴸ_tcommon_center_1, wᵢᵁ_tcommon_center_1])
    ŵᴸ_tcommon_center_1::Matrix{T}, ŵᵁ_tcommon_center_1::Matrix{T},
    Ŵ_tcommon_center_1::Vector{Vector{Interval{T}}}, # ([ŵᵢᴸ, ŵᵢᵁ])
    optimalValue_tcommon_center_1::T,
    } where {T <: Real}

function solvetCommonGroundLP2(matrices::Vector{Matrix{T}})::Union{LPResult_t_2_CommonGround, Nothing} where {T <: Real}
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

    ḋ_tcommon_center_1 = map(Aₖ -> solveCrispAHPLP(Aₖ).optimalValue_center_1, matrices)

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    try
        # wᵢᴸ_tcommon_center_1 ≥ ε, wᵢᵁ_tcommon_center_1 ≥ ε
        @variable(model, wᴸ_tcommon_center_1[i=1:n] ≥ ε); @variable(model, wᵁ_tcommon_center_1[i=1:n] ≥ ε)
        # ŵₖᵢᴸ_tcommon_center_1 ≥ ε, ŵₖᵢᵁ_tcommon_center_1 ≥ ε
        @variable(model, ŵᴸ_tcommon_center_1[k=1:l,i=1:n] ≥ ε); @variable(model, ŵᵁ_tcommon_center_1[k=1:l,i=1:n] ≥ ε)

        for k = 1:l
            ŵₖᴸ_tcommon_center_1 = ŵᴸ_tcommon_center_1[k,:]; ŵₖᵁ_tcommon_center_1 = ŵᵁ_tcommon_center_1[k,:]

            Aₖ = matrices[k]

            # ∑(ŵₖᵢᵁ_tcommon_center_1 - ŵₖᵢᴸ_tcommon_center_1) ≤ sₖḋ_tcommon_center_1ₖ
            @constraint(model, sum(ŵₖᵁ_tcommon_center_1) - sum(ŵₖᴸ_tcommon_center_1) == (sum(ŵₖᴸ_tcommon_center_1) + sum(ŵₖᵁ_tcommon_center_1)) / 2 * ḋ_tcommon_center_1[k])

            for i = 1:n-1
                ŵₖᵢᴸ_tcommon_center_1 = ŵₖᴸ_tcommon_center_1[i]; ŵₖᵢᵁ_tcommon_center_1 = ŵₖᵁ_tcommon_center_1[i]
    
                for j = i+1:n
                    aₖᵢⱼ = Aₖ[i,j]
                    ŵₖⱼᴸ_tcommon_center_1 = ŵₖᴸ_tcommon_center_1[j]; ŵₖⱼᵁ_tcommon_center_1 = ŵₖᵁ_tcommon_center_1[j]
                    
                    @constraint(model, ŵₖᵢᴸ_tcommon_center_1 ≤ aₖᵢⱼ * ŵₖⱼᵁ_tcommon_center_1)
                    @constraint(model, aₖᵢⱼ * ŵₖⱼᴸ_tcommon_center_1 ≤ ŵₖᵢᵁ_tcommon_center_1)
                end
            end

            for i = 1:n
                ŵₖᵢᴸ_tcommon_center_1 = ŵₖᴸ_tcommon_center_1[i]; ŵₖᵢᵁ_tcommon_center_1 = ŵₖᵁ_tcommon_center_1[i]

                # 正規性条件
                ∑ŵₖⱼᴸ_tcommon_center_1 = sum(map(j -> ŵₖᴸ_tcommon_center_1[j], filter(j -> i != j, 1:n)))
                @constraint(model, ∑ŵₖⱼᴸ_tcommon_center_1 + ŵₖᵢᵁ_tcommon_center_1 ≤ 1)
                ∑ŵₖⱼᵁ_tcommon_center_1 = sum(map(j -> ŵₖᵁ_tcommon_center_1[j], filter(j -> i != j, 1:n)))
                @constraint(model, ∑ŵₖⱼᵁ_tcommon_center_1 + ŵₖᵢᴸ_tcommon_center_1 ≥ 1)

                wᵢᴸ_tcommon_center_1 = wᴸ_tcommon_center_1[i]; wᵢᵁ_tcommon_center_1 = wᵁ_tcommon_center_1[i]
                @constraint(model, wᵢᴸ_tcommon_center_1 ≥ ŵₖᵢᴸ_tcommon_center_1)
                @constraint(model, wᵢᵁ_tcommon_center_1 ≥ wᵢᴸ_tcommon_center_1)
                @constraint(model, ŵₖᵢᵁ_tcommon_center_1 ≥ wᵢᵁ_tcommon_center_1)
            end
        end

        @constraint(model, sum(wᵁ_tcommon_center_1) + sum(wᴸ_tcommon_center_1) == 2)

        for i = 1:n
            wᵢᴸ_tcommon_center_1 = wᴸ_tcommon_center_1[i]; wᵢᵁ_tcommon_center_1 = wᵁ_tcommon_center_1[i] 
            # kなしの正規性条件
            ∑wⱼᴸ_tcommon_center_1 = sum(map(j -> wᴸ_tcommon_center_1[j], filter(j -> i != j, 1:n)))
            @constraint(model, ∑wⱼᴸ_tcommon_center_1 + wᵢᵁ_tcommon_center_1 ≤ 1)
            ∑wⱼᵁ_tcommon_center_1 = sum(map(j -> wᵁ_tcommon_center_1[j], filter(j -> i != j, 1:n)))
            @constraint(model, ∑wⱼᵁ_tcommon_center_1 + wᵢᴸ_tcommon_center_1 ≥ 1)
        end

        # 目的関数 ∑(wᵢᵁ_tcommon_center_1 - wᵢᴸ_tcommon_center_1)
        @objective(model, Max, sum(wᵁ_tcommon_center_1) - sum(wᴸ_tcommon_center_1))

        optimize!(model)

        if termination_status(model) == MOI.OPTIMAL
            # 解が見つかった場合の処理
            optimalValue_tcommon_center_1 = sum(value.(wᵁ_tcommon_center_1)) - sum(value.(wᴸ_tcommon_center_1))

            wᴸ_tcommon_center_1_value = value.(wᴸ_tcommon_center_1)
            wᵁ_tcommon_center_1_value = value.(wᵁ_tcommon_center_1)
    
            # precision error 対応
            for i = 1:n
                if wᴸ_tcommon_center_1_value[i] > wᵁ_tcommon_center_1_value[i]
                    wᴸ_tcommon_center_1_value[i] = wᵁ_tcommon_center_1_value[i]
                end
            end
            W_tcommon_center_1_value = map(i -> (wᴸ_tcommon_center_1_value[i])..(wᵁ_tcommon_center_1_value[i]), 1:n)
            
            ŵᴸ_tcommon_center_1_value = value.(ŵᴸ_tcommon_center_1)
            ŵᵁ_tcommon_center_1_value = value.(ŵᵁ_tcommon_center_1)
    
            # precision error 対応
            for k = 1:l, i = 1:n
                if ŵᴸ_tcommon_center_1_value[k,i] > ŵᵁ_tcommon_center_1_value[k,i]
                    ŵᴸ_tcommon_center_1_value[k,i] = ŵᵁ_tcommon_center_1_value[k,i]
                end
            end
            Ŵ_tcommon_center_1_value = map(
                k -> map(i -> (ŵᴸ_tcommon_center_1_value[k,i])..(ŵᵁ_tcommon_center_1_value[k,i]), 1:n),
                1:l)
    
            return (
                wᴸ_tcommon_center_1=wᴸ_tcommon_center_1_value, wᵁ_tcommon_center_1=wᵁ_tcommon_center_1_value,
                W_tcommon_center_1=W_tcommon_center_1_value,
                ŵᴸ_tcommon_center_1=ŵᴸ_tcommon_center_1_value, ŵᵁ_tcommon_center_1=ŵᵁ_tcommon_center_1_value,
                Ŵ_tcommon_center_1=Ŵ_tcommon_center_1_value,
                optimalValue_tcommon_center_1=optimalValue_tcommon_center_1,
            )
        else
            # 解が見つからなかった場合の処理
            println("The tCommonGround optimization problem had no optimal solution.")
            return nothing  # 解が見つからなかったことを示すためにnothingを返す
        end

    catch e
        # エラーが発生した場合の処理
        println("An error occurred during optimization: ", e)
        return nothing  # エラーが発生したことを示すためにnothingを返す
    end
end
