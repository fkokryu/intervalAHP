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
include("perfect-nonunique.jl")
include("common-nonunique.jl")
include("partial-nonunique.jl")
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

function result_weights(method, name)
    # methodが"Error"の場合、エラーを返す
    if method == "Error"
        return "Error"
    else
        # Methodのフィールドにアクセスして計算を行う
        wᴸ = getfield(method, Symbol("wᴸ_", name))
        wᵁ = getfield(method, Symbol("wᵁ_", name))

        # wᴸとwᵁをタプルとして返す
        return (wᴸ, wᵁ)
    end
end

function result_weights_k(method, k, name)
    # methodが"Error"の場合、エラーを返す
    if method == "Error"
        return "Error"
    else
        # Methodのフィールドにアクセスして計算を行う
        Wᴸ = getfield(method, Symbol("ŵᴸ_", name))
        Wᵁ = getfield(method, Symbol("ŵᵁ_", name))
        
        wᴸ_k = Wᴸ[k, :]
        wᵁ_k = Wᵁ[k, :]

        # wᴸとwᵁをタプルとして返す
        return (wᴸ_k, wᵁ_k)
    end
end

# 実験を実行する関数（修正版）
function run_experiments_nonunique(pcms, n, k, trials, result_filename::String, result_filename2::String, pcm_filename::String)
    temp_results = Array{Any}(undef, trials)
    temp_results2 = Array{Any}(undef, trials)
    local_pcms = Array{Any}(undef, trials)
    
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
        local_pcms[i] = (i, selected_pcms, pcm_identifiers, pcm_weights)

         # PerfectIncorporationの計算
        PerfectIncorporation_before = solvePerfectIncorporationLP(selected_pcms)
        if PerfectIncorporation_before === nothing
            PerfectIncorporation_before = "Error"
        end
        PerfectIncorporation = solveonePerfectIncorporationLP(selected_pcms)
        if PerfectIncorporation === nothing
            PerfectIncorporation = "Error"
        end
        tPerfectIncorporation2 = solveNonUniquePerfectIncorporationLP(selected_pcms)
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
        tCommonGround2 = solveNonUniqueCommonGroundLP(selected_pcms)
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
        tPartialIncorporation2 = solveNonUniquePartialIncorporationLP(selected_pcms)
        if tPartialIncorporation2 === nothing
            tPartialIncorporation2 = "Error"
        end

        # スレッドごとの結果を一時的に保存
        local_results = (
            PCM_Identifiers = pcm_identifiers,
            entani論文のPerfectIncorporation = calculate_result(PerfectIncorporation_before, n, "perfect_entani"),
            中心総和1のPerfectIncorporation = calculate_result(PerfectIncorporation, n, "perfect_center_1"),
            解の非唯一性考慮のPerfectIncorporation = calculate_result(tPerfectIncorporation2, n, "perfect_nonunique"),
            entani論文のCommonGround = calculate_result(CommonGround_before, n, "common_entani"),
            中心総和1のCommonGround = calculate_result(CommonGround, n, "common_center_1"),
            解の非唯一性考慮のCommonGround = calculate_result(tCommonGround2, n, "common_nonunique"),
            entani論文のPartialIncorporation = calculate_result(PartialIncorporation_before, n, "partial_entani"),
            中心総和1のPartialIncorporation = calculate_result(PartialIncorporation, n, "partial_center_1"),
            解の非唯一性考慮のPartialIncorporation = calculate_result(tPartialIncorporation2, n, "partial_nonunique"),
        )

        # スレッドごとの結果を一時的に保存
        local_results2 = (
            Trial = i,
            entani論文のPerfectIncorporation_wᴸ = result_weights(PerfectIncorporation_before, "perfect_entani")[1],
            entani論文のPerfectIncorporation_wᵁ = result_weights(PerfectIncorporation_before, "perfect_entani")[2],
            entani論文のPerfectIncorporation_wᴸ_1 = result_weights_k(PerfectIncorporation_before, 1, "perfect_entani")[1],
            entani論文のPerfectIncorporation_wᵁ_1 = result_weights_k(PerfectIncorporation_before, 1, "perfect_entani")[2],
            entani論文のPerfectIncorporation_wᴸ_2 = result_weights_k(PerfectIncorporation_before, 2, "perfect_entani")[1],
            entani論文のPerfectIncorporation_wᵁ_2 = result_weights_k(PerfectIncorporation_before, 2, "perfect_entani")[2],
            entani論文のPerfectIncorporation_wᴸ_3 = result_weights_k(PerfectIncorporation_before, 3, "perfect_entani")[1],
            entani論文のPerfectIncorporation_wᵁ_3 = result_weights_k(PerfectIncorporation_before, 3, "perfect_entani")[2],            
            entani論文のCommonGround_wᴸ = result_weights(CommonGround_before,  "common_entani")[1],
            entani論文のCommonGround_wᵁ = result_weights(CommonGround_before, "common_entani")[2],
            entani論文のCommonGround_wᴸ_1 = result_weights_k(CommonGround_before, 1, "common_entani")[1],
            entani論文のCommonGround_wᵁ_1 = result_weights_k(CommonGround_before, 1, "common_entani")[2],
            entani論文のCommonGround_wᴸ_2 = result_weights_k(CommonGround_before, 2, "common_entani")[1],
            entani論文のCommonGround_wᵁ_2 = result_weights_k(CommonGround_before, 2, "common_entani")[2],
            entani論文のCommonGround_wᴸ_3 = result_weights_k(CommonGround_before, 3, "common_entani")[1],
            entani論文のCommonGround_wᵁ_3 = result_weights_k(CommonGround_before, 3, "common_entani")[2],
            entani論文のPartialIncorporation_wᴸ = result_weights(PartialIncorporation_before, "partial_entani")[1],
            entani論文のPartialIncorporation_wᵁ = result_weights(PartialIncorporation_before, "partial_entani")[2],
            entani論文のPartialIncorporation_wᴸ_1 = result_weights_k(PartialIncorporation_before, 1, "partial_entani")[1],
            entani論文のPartialIncorporation_wᵁ_1 = result_weights_k(PartialIncorporation_before, 1, "partial_entani")[2],
            entani論文のPartialIncorporation_wᴸ_2 = result_weights_k(PartialIncorporation_before, 2, "partial_entani")[1],
            entani論文のPartialIncorporation_wᵁ_2 = result_weights_k(PartialIncorporation_before, 2, "partial_entani")[2],
            entani論文のPartialIncorporation_wᴸ_3 = result_weights_k(PartialIncorporation_before, 3, "partial_entani")[1],
            entani論文のPartialIncorporation_wᵁ_3 = result_weights_k(PartialIncorporation_before, 3, "partial_entani")[2],
            entani論文のPartialIncorporation_Ŵ = getfield(PartialIncorporation_before, :w_partial_entani),
        )

        temp_results[i] = local_results
        temp_results2[i] = local_results2
    end
    
    # 全てのスレッドの結果をall_pcmsに統合
    for result in local_pcms
        push!(all_pcms, result)
    end

    # 1からdesired_countまでの数字を含む配列を作成
    missing_numbers = setdiff(1:desired_count, [x[1] for x in all_pcms])

    # 結果をDataFrameに変換
    results = DataFrame(vcat(temp_results...))
    results2 = DataFrame(vcat(temp_results2...))

    # 結果をCSVファイルに保存
    CSV.write(result_filename, results)
    CSV.write(result_filename2, results2)

    # all_pcmsをトライアル番号に基づいてソート
    sort!(all_pcms, by = x -> x[1])

    # 1からdesired_countまでの数字を含む配列を作成
    missing_numbers = setdiff(1:desired_count, [x[1] for x in all_pcms])

    # 欠けている数字があれば表示
    if isempty(missing_numbers)
        println("全ての数字が存在します。")
    else
        println("欠けている数字: ", join(missing_numbers, ", "))
    end

    # PCMデータと区間重要度をCSVに書き込む
    open(pcm_filename, "w") do file
        # ヘッダーを動的に生成して書き込む
        pcm_header = join(["pcm_$i" for i in 1:n], ",")
        write(file, "Trial,PCM,$pcm_header,,w_L,w_U\n")

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