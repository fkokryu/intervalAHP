using IntervalArithmetic
using JuMP
import HiGHS

include("../crisp-pcm.jl")
include("../nearly-equal.jl")
include("../interval-ahp.jl")

LPResult_CommonGround2 = @NamedTuple{
    # 区間重みベクトル
    wᴸ_common_entani_normal::Vector{T}, wᵁ_common_entani_normal::Vector{T},
    W_common_entani_normal::Vector{Interval{T}}, # ([wᵢᴸ_common_entani_normal, wᵢᵁ_common_entani_normal])
    ŵᴸ_common_entani_normal::Matrix{T}, ŵᵁ_common_entani_normal::Matrix{T},
    Ŵ_common_entani_normal::Vector{Vector{Interval{T}}}, # ([ŵᵢᴸ, ŵᵢᵁ])
    optimalValue_common_entani_normal::T
    } where {T <: Real}

function solveCommonGroundLP2(matrices::Vector{Matrix{T}})::Union{LPResult_CommonGround2{T}, Nothing} where {T <: Real}
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

    ḋ_common_entani_normal = map(Aₖ -> solveIntervalAHPLP(Aₖ).optimalValue, matrices)

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    try
        # wᵢᴸ_common_entani_normal ≥ ε, wᵢᵁ_common_entani_normal ≥ ε
        @variable(model, wᴸ_common_entani_normal[i=1:n] ≥ ε); @variable(model, wᵁ_common_entani_normal[i=1:n] ≥ ε)
        # ŵₖᵢᴸ_common_entani_normal ≥ ε, ŵₖᵢᵁ_common_entani_normal ≥ ε
        @variable(model, ŵᴸ_common_entani_normal[k=1:l,i=1:n] ≥ ε); @variable(model, ŵᵁ_common_entani_normal[k=1:l,i=1:n] ≥ ε)

        for k = 1:l
            ŵₖᴸ_common_entani_normal = ŵᴸ_common_entani_normal[k,:]; ŵₖᵁ_common_entani_normal = ŵᵁ_common_entani_normal[k,:]

            Aₖ = matrices[k]

            # ∑(ŵₖᵢᵁ_common_entani_normal - ŵₖᵢᴸ_common_entani_normal) ≤ ḋ_common_entani_normalₖ
            @constraint(model, sum(ŵₖᵁ_common_entani_normal) - sum(ŵₖᴸ_common_entani_normal) ≤ (ḋ_common_entani_normal[k] + ε))

            for i = 1:n-1
                ŵₖᵢᴸ_common_entani_normal = ŵₖᴸ_common_entani_normal[i]; ŵₖᵢᵁ_common_entani_normal = ŵₖᵁ_common_entani_normal[i]
    
                for j = i+1:n
                    aₖᵢⱼ = Aₖ[i,j]
                    ŵₖⱼᴸ_common_entani_normal = ŵₖᴸ_common_entani_normal[j]; ŵₖⱼᵁ_common_entani_normal = ŵₖᵁ_common_entani_normal[j]
                    
                    @constraint(model, ŵₖᵢᴸ_common_entani_normal ≤ aₖᵢⱼ * ŵₖⱼᵁ_common_entani_normal)
                    @constraint(model, aₖᵢⱼ * ŵₖⱼᴸ_common_entani_normal ≤ ŵₖᵢᵁ_common_entani_normal)
                end
            end

            for i = 1:n
                ŵₖᵢᴸ_common_entani_normal = ŵₖᴸ_common_entani_normal[i]; ŵₖᵢᵁ_common_entani_normal = ŵₖᵁ_common_entani_normal[i]

                # 正規性条件
                ∑ŵₖⱼᴸ_common_entani_normal = sum(map(j -> ŵₖᴸ_common_entani_normal[j], filter(j -> i != j, 1:n)))
                @constraint(model, ∑ŵₖⱼᴸ_common_entani_normal + ŵₖᵢᵁ_common_entani_normal ≤ 1)
                ∑ŵₖⱼᵁ_common_entani_normal = sum(map(j -> ŵₖᵁ_common_entani_normal[j], filter(j -> i != j, 1:n)))
                @constraint(model, ∑ŵₖⱼᵁ_common_entani_normal + ŵₖᵢᴸ_common_entani_normal ≥ 1)

                wᵢᴸ_common_entani_normal = wᴸ_common_entani_normal[i]; wᵢᵁ_common_entani_normal = wᵁ_common_entani_normal[i]
                @constraint(model, wᵢᴸ_common_entani_normal ≥ ŵₖᵢᴸ_common_entani_normal)
                @constraint(model, wᵢᵁ_common_entani_normal ≥ wᵢᴸ_common_entani_normal)
                @constraint(model, ŵₖᵢᵁ_common_entani_normal ≥ wᵢᵁ_common_entani_normal)
            end
        end

        for i = 1:n
            wᵢᴸ_common_entani_normal = wᴸ_common_entani_normal[i]; wᵢᵁ_common_entani_normal = wᵁ_common_entani_normal[i] 
            # kなしの正規性条件
            ∑wⱼᴸ_common_entani_normal = sum(map(j -> wᴸ_common_entani_normal[j], filter(j -> i != j, 1:n)))
            @constraint(model, ∑wⱼᴸ_common_entani_normal + wᵢᵁ_common_entani_normal ≤ 1)
            ∑wⱼᵁ_common_entani_normal = sum(map(j -> wᵁ_common_entani_normal[j], filter(j -> i != j, 1:n)))
            @constraint(model, ∑wⱼᵁ_common_entani_normal + wᵢᴸ_common_entani_normal ≥ 1)
        end

        @constraint(model, sum(wᵁ_common_entani_normal) + sum(wᴸ_common_entani_normal) == 2)

        # 目的関数 ∑(wᵢᵁ_common_entani_normal - wᵢᴸ_common_entani_normal)
        @objective(model, Max, sum(wᵁ_common_entani_normal) - sum(wᴸ_common_entani_normal))

        optimize!(model)

        if termination_status(model) == MOI.OPTIMAL
            # 解が見つかった場合の処理
            optimalValue_common_entani_normal = sum(value.(wᵁ_common_entani_normal)) - sum(value.(wᴸ_common_entani_normal))

            wᴸ_common_entani_normal_value = value.(wᴸ_common_entani_normal)
            wᵁ_common_entani_normal_value = value.(wᵁ_common_entani_normal)
            # precision error 対応
            for i = 1:n
                if wᴸ_common_entani_normal_value[i] > wᵁ_common_entani_normal_value[i]
                    wᴸ_common_entani_normal_value[i] = wᵁ_common_entani_normal_value[i]
                end
            end
            W_common_entani_normal_value = map(i -> (wᴸ_common_entani_normal_value[i])..(wᵁ_common_entani_normal_value[i]), 1:n)
            
            ŵᴸ_common_entani_normal_value = value.(ŵᴸ_common_entani_normal)
            ŵᵁ_common_entani_normal_value = value.(ŵᵁ_common_entani_normal)
            # precision error 対応
            for k = 1:l, i = 1:n
                if ŵᴸ_common_entani_normal_value[k,i] > ŵᵁ_common_entani_normal_value[k,i]
                    ŵᴸ_common_entani_normal_value[k,i] = ŵᵁ_common_entani_normal_value[k,i]
                end
            end
            Ŵ_common_entani_normal_value = map(
                k -> map(i -> (ŵᴸ_common_entani_normal_value[k,i])..(ŵᵁ_common_entani_normal_value[k,i]), 1:n),
                1:l)

            return (
                wᴸ_common_entani_normal=wᴸ_common_entani_normal_value, wᵁ_common_entani_normal=wᵁ_common_entani_normal_value,
                W_common_entani_normal=W_common_entani_normal_value,
                ŵᴸ_common_entani_normal=ŵᴸ_common_entani_normal_value, ŵᵁ_common_entani_normal=ŵᵁ_common_entani_normal_value,
                Ŵ_common_entani_normal=Ŵ_common_entani_normal_value,
                optimalValue_common_entani_normal=optimalValue_common_entani_normal
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
