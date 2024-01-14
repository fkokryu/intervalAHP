using IntervalArithmetic
using JuMP
import HiGHS

include("../crisp-pcm.jl")
include("../nearly-equal.jl")
include("../ttimes/optimal-value.jl")

LPResult_one_PerfectIncorporation = @NamedTuple{
    # 区間重みベクトル
    wᴸ_perfect_center_1::Vector{T}, wᵁ_perfect_center_1::Vector{T},
    W_perfect_center_1::Vector{Interval{T}}, # ([wᵢᴸ_perfect_center_1, wᵢᵁ_perfect_center_1])
    ŵᴸ_perfect_center_1::Matrix{T}, ŵᵁ_perfect_center_1::Matrix{T},
    Ŵ_perfect_center_1::Vector{Vector{Interval{T}}}, # ([ŵᵢᴸ, ŵᵢᵁ])
    optimalValue_perfect_center_1::T
    } where {T <: Real}

function solveonePerfectIncorporationLP(matrices::Vector{Matrix{T}})::Union{LPResult_one_PerfectIncorporation{T}, Nothing} where {T <: Real}
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

    ḋ = map(Aₖ -> solveCrispAHPLP(Aₖ).optimalValue_center_1, matrices)

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    try
        # wᵢᴸ_perfect_center_1 ≥ ε, wᵢᵁ_perfect_center_1 ≥ ε
        @variable(model, wᴸ_perfect_center_1[i=1:n] ≥ ε); @variable(model, wᵁ_perfect_center_1[i=1:n] ≥ ε)
        # ŵₖᵢᴸ_perfect_center_1 ≥ ε, ŵₖᵢᵁ_perfect_center_1 ≥ ε
        @variable(model, ŵᴸ_perfect_center_1[k=1:l,i=1:n] ≥ ε); @variable(model, ŵᵁ_perfect_center_1[k=1:l,i=1:n] ≥ ε)

        for k = 1:l
            ŵₖᴸ_perfect_center_1 = ŵᴸ_perfect_center_1[k,:]; ŵₖᵁ_perfect_center_1 = ŵᵁ_perfect_center_1[k,:]

            Aₖ = matrices[k]

            # ∑(ŵₖᵢᵁ_perfect_center_1 - ŵₖᵢᴸ_perfect_center_1) ≤ ḋₖ
            @constraint(model, sum(ŵₖᵁ_perfect_center_1) - sum(ŵₖᴸ_perfect_center_1) ≤ ḋ[k])

            for i = 1:n-1
                ŵₖᵢᴸ_perfect_center_1 = ŵₖᴸ_perfect_center_1[i]; ŵₖᵢᵁ_perfect_center_1 = ŵₖᵁ_perfect_center_1[i]
    
                for j = i+1:n
                    aₖᵢⱼ = Aₖ[i,j]
                    ŵₖⱼᴸ_perfect_center_1 = ŵₖᴸ_perfect_center_1[j]; ŵₖⱼᵁ_perfect_center_1 = ŵₖᵁ_perfect_center_1[j]
                    
                    @constraint(model, ŵₖᵢᴸ_perfect_center_1 ≤ aₖᵢⱼ * ŵₖⱼᵁ_perfect_center_1)
                    @constraint(model, aₖᵢⱼ * ŵₖⱼᴸ_perfect_center_1 ≤ ŵₖᵢᵁ_perfect_center_1)
                end
            end

            for i = 1:n
                ŵₖᵢᴸ_perfect_center_1 = ŵₖᴸ_perfect_center_1[i]; ŵₖᵢᵁ_perfect_center_1 = ŵₖᵁ_perfect_center_1[i]

                # 正規性条件
                ∑ŵₖⱼᴸ_perfect_center_1 = sum(map(j -> ŵₖᴸ_perfect_center_1[j], filter(j -> i != j, 1:n)))
                @constraint(model, ∑ŵₖⱼᴸ_perfect_center_1 + ŵₖᵢᵁ_perfect_center_1 ≤ 1)
                ∑ŵₖⱼᵁ_perfect_center_1 = sum(map(j -> ŵₖᵁ_perfect_center_1[j], filter(j -> i != j, 1:n)))
                @constraint(model, ∑ŵₖⱼᵁ_perfect_center_1 + ŵₖᵢᴸ_perfect_center_1 ≥ 1)

                wᵢᴸ_perfect_center_1 = wᴸ_perfect_center_1[i]; wᵢᵁ_perfect_center_1 = wᵁ_perfect_center_1[i] 
                @constraint(model, ŵₖᵢᴸ_perfect_center_1 ≥ wᵢᴸ_perfect_center_1)
                @constraint(model, ŵₖᵢᵁ_perfect_center_1 ≥ ŵₖᵢᴸ_perfect_center_1)
                @constraint(model, wᵢᵁ_perfect_center_1 ≥ ŵₖᵢᵁ_perfect_center_1)
            end
            @constraint(model, sum(ŵₖᴸ_perfect_center_1) + sum(ŵₖᵁ_perfect_center_1) == 2)
        end

        @constraint(model, sum(wᵁ_perfect_center_1) + sum(wᴸ_perfect_center_1) == 2)

        # 目的関数 ∑(wᵢᵁ_perfect_center_1 - wᵢᴸ_perfect_center_1)
        @objective(model, Min, sum(wᵁ_perfect_center_1) - sum(wᴸ_perfect_center_1))

        optimize!(model)

        if termination_status(model) == MOI.OPTIMAL
            # 解が見つかった場合の処理
            optimalValue_perfect_center_1 = sum(value.(wᵁ_perfect_center_1)) - sum(value.(wᴸ_perfect_center_1))

            wᴸ_perfect_center_1_value = value.(wᴸ_perfect_center_1)
            wᵁ_perfect_center_1_value = value.(wᵁ_perfect_center_1)
            # precision error 対応
            for i = 1:n
                if wᴸ_perfect_center_1_value[i] > wᵁ_perfect_center_1_value[i]
                    wᴸ_perfect_center_1_value[i] = wᵁ_perfect_center_1_value[i]
                end
            end
            W_perfect_center_1_value = map(i -> (wᴸ_perfect_center_1_value[i])..(wᵁ_perfect_center_1_value[i]), 1:n)
            
            ŵᴸ_perfect_center_1_value = value.(ŵᴸ_perfect_center_1)
            ŵᵁ_perfect_center_1_value = value.(ŵᵁ_perfect_center_1)
            # precision error 対応
            for k = 1:l, i = 1:n
                if ŵᴸ_perfect_center_1_value[k,i] > ŵᵁ_perfect_center_1_value[k,i]
                    ŵᴸ_perfect_center_1_value[k,i] = ŵᵁ_perfect_center_1_value[k,i]
                end
            end
            Ŵ_perfect_center_1_value = map(
                k -> map(i -> (ŵᴸ_perfect_center_1_value[k,i])..(ŵᵁ_perfect_center_1_value[k,i]), 1:n),
                1:l)

            return (
                wᴸ_perfect_center_1=wᴸ_perfect_center_1_value, wᵁ_perfect_center_1=wᵁ_perfect_center_1_value,
                W_perfect_center_1=W_perfect_center_1_value,
                ŵᴸ_perfect_center_1=ŵᴸ_perfect_center_1_value, ŵᵁ_perfect_center_1=ŵᵁ_perfect_center_1_value,
                Ŵ_perfect_center_1=Ŵ_perfect_center_1_value,
                optimalValue_perfect_center_1=optimalValue_perfect_center_1
            )
        else
            # 解が見つからなかった場合の処理
            println("The PerfectIncorporation optimization problem had no optimal solution.")
            return nothing  # 解が見つからなかったことを示すためにnothingを返す
        end

    catch e
        # エラーが発生した場合の処理
        println("An error occurred during optimization: ", e)
        return nothing  # エラーが発生したことを示すためにnothingを返す
    end
end
