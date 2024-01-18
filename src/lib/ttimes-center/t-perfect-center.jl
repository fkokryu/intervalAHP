using IntervalArithmetic
using JuMP
import HiGHS

include("../crisp-pcm.jl")
include("../nearly-equal.jl")
include("../ttimes/optimal-value.jl")

LPResult_t_2_PerfectIncorporation = @NamedTuple{
    # 区間重みベクトル
    wᴸ_tperfect_center_1::Vector{T}, wᵁ_tperfect_center_1::Vector{T},
    W_tperfect_center_1::Vector{Interval{T}}, # ([wᵢᴸ_tperfect_center_1, wᵢᵁ_tperfect_center_1])
    ŵᴸ_tperfect_center_1::Matrix{T}, ŵᵁ_tperfect_center_1::Matrix{T},
    Ŵ_tperfect_center_1::Vector{Vector{Interval{T}}}, # ([ŵᵢᴸ, ŵᵢᵁ])
    optimalValue_tperfect_center_1::T,
    s_tperfect_center_1::T
    } where {T <: Real}

function solvetPerfectIncorporationLP2(matrices::Vector{Matrix{T}})::Union{LPResult_t_2_PerfectIncorporation{T}, Nothing} where {T <: Real}
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

    ḋ_tperfect_center_1 = map(Aₖ -> solveCrispAHPLP(Aₖ).optimalValue_center_1, matrices)

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    try
        # wᵢᴸ_tperfect_center_1 ≥ ε, wᵢᵁ_tperfect_center_1 ≥ ε
        @variable(model, wᴸ_tperfect_center_1[i=1:n] ≥ ε); @variable(model, wᵁ_tperfect_center_1[i=1:n] ≥ ε)
        # ŵₖᵢᴸ_tperfect_center_1 ≥ ε, ŵₖᵢᵁ_tperfect_center_1 ≥ ε
        @variable(model, ŵᴸ_tperfect_center_1[k=1:l,i=1:n] ≥ ε); @variable(model, ŵᵁ_tperfect_center_1[k=1:l,i=1:n] ≥ ε)

        @variable(model, s_tperfect_center_1 ≥ ε)

        for k = 1:l
            ŵₖᴸ_tperfect_center_1 = ŵᴸ_tperfect_center_1[k,:]; ŵₖᵁ_tperfect_center_1 = ŵᵁ_tperfect_center_1[k,:]

            Aₖ = matrices[k]

            # ∑(ŵₖᵢᵁ_tperfect_center_1 - ŵₖᵢᴸ_tperfect_center_1) ≤ sₖḋ_tperfect_center_1ₖ
            @constraint(model, sum(ŵₖᵁ_tperfect_center_1) - sum(ŵₖᴸ_tperfect_center_1) ≤ ((sum(ŵₖᴸ_tperfect_center_1) + sum(ŵₖᵁ_tperfect_center_1)) / 2 * (ḋ_tperfect_center_1[k]) + ε) )

            for i = 1:n-1
                ŵₖᵢᴸ_tperfect_center_1 = ŵₖᴸ_tperfect_center_1[i]; ŵₖᵢᵁ_tperfect_center_1 = ŵₖᵁ_tperfect_center_1[i]
    
                for j = i+1:n
                    aₖᵢⱼ = Aₖ[i,j]
                    ŵₖⱼᴸ_tperfect_center_1 = ŵₖᴸ_tperfect_center_1[j]; ŵₖⱼᵁ_tperfect_center_1 = ŵₖᵁ_tperfect_center_1[j]
                    
                    @constraint(model, ŵₖᵢᴸ_tperfect_center_1 ≤ aₖᵢⱼ * ŵₖⱼᵁ_tperfect_center_1)
                    @constraint(model, aₖᵢⱼ * ŵₖⱼᴸ_tperfect_center_1 ≤ ŵₖᵢᵁ_tperfect_center_1)
                end
            end

            for i = 1:n
                ŵₖᵢᴸ_tperfect_center_1 = ŵₖᴸ_tperfect_center_1[i]; ŵₖᵢᵁ_tperfect_center_1 = ŵₖᵁ_tperfect_center_1[i]

                # 正規性条件
                ∑ŵₖⱼᴸ_tperfect_center_1 = sum(map(j -> ŵₖᴸ_tperfect_center_1[j], filter(j -> i != j, 1:n)))
                @constraint(model, ∑ŵₖⱼᴸ_tperfect_center_1 + ŵₖᵢᵁ_tperfect_center_1 ≤ s_tperfect_center_1)
                ∑ŵₖⱼᵁ_tperfect_center_1 = sum(map(j -> ŵₖᵁ_tperfect_center_1[j], filter(j -> i != j, 1:n)))
                @constraint(model, ∑ŵₖⱼᵁ_tperfect_center_1 + ŵₖᵢᴸ_tperfect_center_1 ≥ s_tperfect_center_1)

                wᵢᴸ_tperfect_center_1 = wᴸ_tperfect_center_1[i]; wᵢᵁ_tperfect_center_1 = wᵁ_tperfect_center_1[i] 
                @constraint(model, ŵₖᵢᴸ_tperfect_center_1 ≥ wᵢᴸ_tperfect_center_1)
                @constraint(model, ŵₖᵢᵁ_tperfect_center_1 ≥ ŵₖᵢᴸ_tperfect_center_1)
                @constraint(model, wᵢᵁ_tperfect_center_1 ≥ ŵₖᵢᵁ_tperfect_center_1)
            end
        end

        @constraint(model, sum(wᵁ_tperfect_center_1) + sum(wᴸ_tperfect_center_1) == 2)

        # 目的関数 ∑(wᵢᵁ_tperfect_center_1 - wᵢᴸ_tperfect_center_1)
        @objective(model, Min, sum(wᵁ_tperfect_center_1) - sum(wᴸ_tperfect_center_1))

        optimize!(model)

        if termination_status(model) == MOI.OPTIMAL
            # 解が見つかった場合の処理
            optimalValue_tperfect_center_1 = sum(value.(wᵁ_tperfect_center_1)) - sum(value.(wᴸ_tperfect_center_1))

            wᴸ_tperfect_center_1_value = value.(wᴸ_tperfect_center_1)
            wᵁ_tperfect_center_1_value = value.(wᵁ_tperfect_center_1)

            # precision error 対応
            for i = 1:n
                if wᴸ_tperfect_center_1_value[i] > wᵁ_tperfect_center_1_value[i]
                    wᴸ_tperfect_center_1_value[i] = wᵁ_tperfect_center_1_value[i]
                end
            end
            W_tperfect_center_1_value = map(i -> (wᴸ_tperfect_center_1_value[i])..(wᵁ_tperfect_center_1_value[i]), 1:n)
            
            ŵᴸ_tperfect_center_1_value = value.(ŵᴸ_tperfect_center_1)
            ŵᵁ_tperfect_center_1_value = value.(ŵᵁ_tperfect_center_1)

            # precision error 対応
            for k = 1:l, i = 1:n
                if ŵᴸ_tperfect_center_1_value[k,i] > ŵᵁ_tperfect_center_1_value[k,i]
                    ŵᴸ_tperfect_center_1_value[k,i] = ŵᵁ_tperfect_center_1_value[k,i]
                end
            end
            Ŵ_tperfect_center_1_value = map(
                k -> map(i -> (ŵᴸ_tperfect_center_1_value[k,i])..(ŵᵁ_tperfect_center_1_value[k,i]), 1:n),
                1:l)
            
            s_tperfect_center_1_value = value.(s_tperfect_center_1)

            return (
                wᴸ_tperfect_center_1=wᴸ_tperfect_center_1_value, wᵁ_tperfect_center_1=wᵁ_tperfect_center_1_value,
                W_tperfect_center_1=W_tperfect_center_1_value,
                ŵᴸ_tperfect_center_1=ŵᴸ_tperfect_center_1_value, ŵᵁ_tperfect_center_1=ŵᵁ_tperfect_center_1_value,
                Ŵ_tperfect_center_1=Ŵ_tperfect_center_1_value,
                optimalValue_tperfect_center_1=optimalValue_tperfect_center_1,
                s_tperfect_center_1=s_tperfect_center_1_value
            )
        else
            # 解が見つからなかった場合の処理
            println("The tPerfectIncorporation2 optimization problem had no optimal solution.")
            return nothing  # 解が見つからなかったことを示すためにnothingを返す
        end

    catch e
        # エラーが発生した場合の処理
        println("An error occurred during optimization: ", e)
        return nothing  # エラーが発生したことを示すためにnothingを返す
    end
end
