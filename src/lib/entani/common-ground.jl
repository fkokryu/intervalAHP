using IntervalArithmetic
using JuMP
import HiGHS

include("../crisp-pcm.jl")
include("../nearly-equal.jl")
include("../interval-ahp.jl")

LPResult_CommonGround = @NamedTuple{
    # 区間重みベクトル
    wᴸ_common_entani::Vector{T}, wᵁ_common_entani::Vector{T},
    W_common_entani::Vector{Interval{T}}, # ([wᵢᴸ_common_entani, wᵢᵁ_common_entani])
    ŵᴸ_common_entani::Matrix{T}, ŵᵁ_common_entani::Matrix{T},
    Ŵ_common_entani::Vector{Vector{Interval{T}}}, # ([ŵᵢᴸ, ŵᵢᵁ])
    optimalValue_common_entani::T
    } where {T <: Real}

function solveCommonGroundLP(matrices::Vector{Matrix{T}})::Union{LPResult_CommonGround{T}, Nothing} where {T <: Real}
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

    ḋ_common_entani = map(Aₖ -> solveIntervalAHPLP(Aₖ).optimalValue, matrices)

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    try
        # wᵢᴸ_common_entani ≥ ε, wᵢᵁ_common_entani ≥ ε
        @variable(model, wᴸ_common_entani[i=1:n] ≥ ε); @variable(model, wᵁ_common_entani[i=1:n] ≥ ε)
        # ŵₖᵢᴸ_common_entani ≥ ε, ŵₖᵢᵁ_common_entani ≥ ε
        @variable(model, ŵᴸ_common_entani[k=1:l,i=1:n] ≥ ε); @variable(model, ŵᵁ_common_entani[k=1:l,i=1:n] ≥ ε)

        for k = 1:l
            ŵₖᴸ_common_entani = ŵᴸ_common_entani[k,:]; ŵₖᵁ_common_entani = ŵᵁ_common_entani[k,:]

            Aₖ = matrices[k]

            # ∑(ŵₖᵢᵁ_common_entani - ŵₖᵢᴸ_common_entani) ≤ ḋ_common_entaniₖ
            @constraint(model, sum(ŵₖᵁ_common_entani) - sum(ŵₖᴸ_common_entani) ≤ (ḋ_common_entani[k] + ε))

            for i = 1:n-1
                ŵₖᵢᴸ_common_entani = ŵₖᴸ_common_entani[i]; ŵₖᵢᵁ_common_entani = ŵₖᵁ_common_entani[i]
    
                for j = i+1:n
                    aₖᵢⱼ = Aₖ[i,j]
                    ŵₖⱼᴸ_common_entani = ŵₖᴸ_common_entani[j]; ŵₖⱼᵁ_common_entani = ŵₖᵁ_common_entani[j]
                    
                    @constraint(model, ŵₖᵢᴸ_common_entani ≤ aₖᵢⱼ * ŵₖⱼᵁ_common_entani)
                    @constraint(model, aₖᵢⱼ * ŵₖⱼᴸ_common_entani ≤ ŵₖᵢᵁ_common_entani)
                end
            end

            for i = 1:n
                ŵₖᵢᴸ_common_entani = ŵₖᴸ_common_entani[i]; ŵₖᵢᵁ_common_entani = ŵₖᵁ_common_entani[i]

                # 正規性条件
                ∑ŵₖⱼᴸ_common_entani = sum(map(j -> ŵₖᴸ_common_entani[j], filter(j -> i != j, 1:n)))
                @constraint(model, ∑ŵₖⱼᴸ_common_entani + ŵₖᵢᵁ_common_entani ≤ 1)
                ∑ŵₖⱼᵁ_common_entani = sum(map(j -> ŵₖᵁ_common_entani[j], filter(j -> i != j, 1:n)))
                @constraint(model, ∑ŵₖⱼᵁ_common_entani + ŵₖᵢᴸ_common_entani ≥ 1)

                wᵢᴸ_common_entani = wᴸ_common_entani[i]; wᵢᵁ_common_entani = wᵁ_common_entani[i]
                @constraint(model, wᵢᴸ_common_entani ≥ ŵₖᵢᴸ_common_entani)
                @constraint(model, wᵢᵁ_common_entani ≥ wᵢᴸ_common_entani)
                @constraint(model, ŵₖᵢᵁ_common_entani ≥ wᵢᵁ_common_entani)
            end
        end

        for i = 1:n
            wᵢᴸ_common_entani = wᴸ_common_entani[i]; wᵢᵁ_common_entani = wᵁ_common_entani[i] 
            # kなしの正規性条件
            ∑wⱼᴸ_common_entani = sum(map(j -> wᴸ_common_entani[j], filter(j -> i != j, 1:n)))
            @constraint(model, ∑wⱼᴸ_common_entani + wᵢᵁ_common_entani ≤ 1)
            ∑wⱼᵁ_common_entani = sum(map(j -> wᵁ_common_entani[j], filter(j -> i != j, 1:n)))
            @constraint(model, ∑wⱼᵁ_common_entani + wᵢᴸ_common_entani ≥ 1)
        end

        # 目的関数 ∑(wᵢᵁ_common_entani - wᵢᴸ_common_entani)
        @objective(model, Max, sum(wᵁ_common_entani) - sum(wᴸ_common_entani))

        optimize!(model)

        if termination_status(model) == MOI.OPTIMAL
            # 解が見つかった場合の処理
            optimalValue_common_entani = sum(value.(wᵁ_common_entani)) - sum(value.(wᴸ_common_entani))

            wᴸ_common_entani_value = value.(wᴸ_common_entani)
            wᵁ_common_entani_value = value.(wᵁ_common_entani)
            # precision error 対応
            for i = 1:n
                if wᴸ_common_entani_value[i] > wᵁ_common_entani_value[i]
                    wᴸ_common_entani_value[i] = wᵁ_common_entani_value[i]
                end
            end
            W_common_entani_value = map(i -> (wᴸ_common_entani_value[i])..(wᵁ_common_entani_value[i]), 1:n)
            
            ŵᴸ_common_entani_value = value.(ŵᴸ_common_entani)
            ŵᵁ_common_entani_value = value.(ŵᵁ_common_entani)
            # precision error 対応
            for k = 1:l, i = 1:n
                if ŵᴸ_common_entani_value[k,i] > ŵᵁ_common_entani_value[k,i]
                    ŵᴸ_common_entani_value[k,i] = ŵᵁ_common_entani_value[k,i]
                end
            end
            Ŵ_common_entani_value = map(
                k -> map(i -> (ŵᴸ_common_entani_value[k,i])..(ŵᵁ_common_entani_value[k,i]), 1:n),
                1:l)

            return (
                wᴸ_common_entani=wᴸ_common_entani_value, wᵁ_common_entani=wᵁ_common_entani_value,
                W_common_entani=W_common_entani_value,
                ŵᴸ_common_entani=ŵᴸ_common_entani_value, ŵᵁ_common_entani=ŵᵁ_common_entani_value,
                Ŵ_common_entani=Ŵ_common_entani_value,
                optimalValue_common_entani=optimalValue_common_entani
            )
        else
            # 解が見つからなかった場合の処理
            println("The CommonGround_entani optimization problem had no optimal solution.")
            return nothing  # 解が見つからなかったことを示すためにnothingを返す
        end

    catch e
        # エラーが発生した場合の処理
        println("An error occurred during optimization: ", e)
        return nothing  # エラーが発生したことを示すためにnothingを返す
    end
end
