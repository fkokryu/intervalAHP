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
include("../interval-ahp.jl")

using Random
using CSV
using DataFrames
using Base.Threads

# ランダムにPCMを選択する関数
function random_select_pcms(pcms, count)
    return pcms[shuffle(1:length(pcms))[1:count]]
end

# 計算結果のチェックと計算を行う関数
function calculate_result(method, n, name)
    # methodが"Error"の場合、エラーを返す
    if method == "Error"
        return "Error"
    else
        # Methodのフィールドにアクセスして計算を行う
        wᴸ = getfield(method, Symbol("wᴸ_", name))
        wᵁ = getfield(method, Symbol("wᵁ_", name))

        return (n - 1) * sum(log.(wᵁ[i]) - log.(wᴸ[i]) for i in 1:n)
    end
end

# 実験を実行する関数（修正版）
function run_experiments(pcms, n, k, trials, result_filename::String, pcm_filename::String)
    temp_results = Array{Any}(undef, trials)
    all_pcms = []  # 全スレッドのPCMデータを保持する配列

    # trials回の実験を実行
    Threads.@threads for i in 1:trials
        selected_pcms = random_select_pcms(pcms, k)

        # PCMデータと識別子を取得
        pcm_identifiers = []
        pcm_weights = []

        for (index, pcm) in enumerate(selected_pcms)
            W = solveIntervalAHPLP(pcm)
            push!(pcm_identifiers, "Trial-$(i)_PCM-$(index)")
            push!(pcm_weights, (W.wᴸ, W.wᵁ))
        end
        push!(all_pcms, (i, selected_pcms, pcm_identifiers, pcm_weights))

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
            PCM_Identifiers = pcm_identifiers,
            entani論文のPerfectIncorporation = calculate_result(PerfectIncorporation_before, n, "perfect_entani"),
            中心総和1のPerfectIncorporation = calculate_result(PerfectIncorporation, n, "perfect_center_1"),
            解の非唯一性考慮のPerfectIncorporation = calculate_result(tPerfectIncorporation2, n, "tperfect_center_1"),
            entani論文のCommonGround = calculate_result(CommonGround_before, n, "common_entani"),
            中心総和1のCommonGround = calculate_result(CommonGround, n, "common_center_1"),
            解の非唯一性考慮のCommonGround = calculate_result(tCommonGround2, n, "tcommon_center_1"),
            entani論文のPartialIncorporation = calculate_result(PartialIncorporation_before, n, "partial_entani"),
            中心総和1のPartialIncorporation = calculate_result(PartialIncorporation, n, "partial_center_1"),
            解の非唯一性考慮のPartialIncorporation = calculate_result(tPartialIncorporation2, n, "tpartial_center_1"),
        )

        temp_results[i] = local_results
    end

    # 結果をDataFrameに変換
    results = DataFrame(vcat(temp_results...))

    # 結果をCSVファイルに保存
    CSV.write(result_filename, results)

    # トライアル番号に基づいてall_pcms配列をソート
    sort!(all_pcms, by = x -> x[1])

    # PCMデータと区間重要度をCSVに書き込む
    open(pcm_filename, "w") do file
        # ヘッダーを書き込む
        write(file, "Trial,PCM,pcm_1,pcm_2,pcm_3,pcm_4,pcm_5,,w_L,w_U\n")

        for (trial_number, pcms, identifiers, weights) in all_pcms
            for (index, (pcm, (wᴸ, wᵁ))) in enumerate(zip(pcms, weights))
                for (row_index, row) in enumerate(eachrow(pcm))
                    # 最初の行には試行回数とPCMインデックスを書き込む
                    
                     write(file, "$trial_number,$index,")
                   
                    # PCMデータを書き込む
                    write(file, join(row, ","))

                    # 対応する区間重要度の左端と右端の値を追加
                    left = wᴸ[row_index]
                    right = wᵁ[row_index]
                    write(file, ",,$left,$right")
                    write(file, "\n")
                end
                # 各PCMデータセットの間に空行を挿入
                write(file, "\n")
            end
        end
    end
end