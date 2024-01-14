# 実験を実行するためのパラメータ
# pcms: PCMのリスト
# n: PCMのサイズ
# k: 選択するPCMの数
# trials: 実験の回数

include("../entani/perfect-incorporation.jl")
include("../entani/common-ground.jl")
include("../entani/partial-incorporarion.jl")
include("../center-equal-one/one-perfect.jl")
include("../center-equal-one/one-common.jl")
include("../center-equal-one/one-partial.jl")
include("../ttimes-center/t-perfect-center.jl")
include("../ttimes-center/t-common-center.jl")
include("../ttimes-center/t-partial-center.jl")
include("../ttimes/optimal-value.jl")

using Random
using CSV
using DataFrames
using Base.Threads

# ランダムにPCMを選択する関数
function random_select_pcms(pcms, count)
    return pcms[shuffle(1:length(pcms))[1:count]]
end

# 計算結果のチェックと計算を行う関数
function calculate_result(Method, n)
    if Method == "Error"
        return "Error"
    else
        return (n-1) * sum(log.(Method.wᵁ[i]) - log.(Method.wᴸ[i]) for i in 1:n)
    end
end

# 実験を実行する関数
function run_experiments(pcms, n, k, trials)    
    # 結果を保存するための配列を初期化
    temp_results = Array{Any}(undef, trials)

    # trials回の実験を実行
    Threads.@threads for i in 1:trials
        selected_pcms = random_select_pcms(pcms, k)

    # PerfectIncorporationの計算
    PerfectIncorporation_before = solvePerfectIncorporationLP(selected_pcms)
    if PerfectIncorporation_before === nothing
        PerfectIncorporation_before = "Error"
    end
    PerfectIncorporation = solveonePerfectIncorporationLP(selected_pcms)
    if PerfectIncorporation === nothing
        PerfectIncorporation = "Error"
    end
    tPerfectIncorporation2 = solvetPerfectIncorporationLP2(selected_pcms)
    if tPerfectIncorporation2 === nothing
        tPerfectIncorporation2 = "Error"
    end

    # CommonGroundの計算
    CommonGround_before = solveCommonGroundLP(selected_pcms)
    if CommonGround_before === nothing
        CommonGround_before = "Error"
    end
    CommonGround = solveoneCommonGroundLP(selected_pcms)
    if CommonGround === nothing
        CommonGround = "Error"
    end
    tCommonGround2 = solvetCommonGroundLP2(selected_pcms)
    if tCommonGround2 === nothing
        tCommonGround2 = "Error"
    end

    # PartialIncorporationの計算
    PartialIncorporation_before = solvePartialIncorporationLP(selected_pcms)
    if PartialIncorporation_before === nothing
        PartialIncorporation_before = "Error"
    end
    PartialIncorporation = solveonePartialIncorporationLP(selected_pcms)
    if PartialIncorporation === nothing
        PartialIncorporation = "Error"
    end
    tPartialIncorporation2 = solvetPartialIncorporationLP2(selected_pcms)
    if tPartialIncorporation2 === nothing
        tPartialIncorporation2 = "Error"
    end

    # スレッドごとの結果を一時的に保存
    local_results = (
        entani論文のPerfectIncorporation = calculate_result(PerfectIncorporation_before, n),
        中心総和1のPerfectIncorporation = calculate_result(PerfectIncorporation, n),
        解の非唯一性考慮のPerfectIncorporation = calculate_result(tPerfectIncorporation2, n),
        entani論文のCommonGround = calculate_result(CommonGround_before, n),
        中心総和1のCommonGround = calculate_result(CommonGround, n),
        解の非唯一性考慮のCommonGround = calculate_result(tCommonGround2, n),
        entani論文のPartialIncorporation = calculate_result(PartialIncorporation_before, n),
        中心総和1のPartialIncorporation = calculate_result(PartialIncorporation, n),
        解の非唯一性考慮のPartialIncorporation = calculate_result(tPartialIncorporation2, n),
    )

        # 結果を一時配列に保存
        temp_results[i] = local_results
    end

    # 結果をDataFrameに変換
    results = DataFrame(vcat(temp_results...))

    # 結果をCSVファイルに保存
    CSV.write("results.csv", results)
end