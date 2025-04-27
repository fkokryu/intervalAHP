# 実験を実行するためのパラメータ
# pcms: PCMのリスト
# n: PCMのサイズ
# k: 選択するPCMの数
# trials: 実験の回数

include("../entani/perfect-incorporation.jl")
include("../entani/common-ground.jl")
include("../entani/partial-incorporarion.jl")
include("../center-equal-one/one-perfect.jl")
include("../entani/common-ground_normal.jl")
include("../center-equal-one/one-partial.jl")
include("../ttimes-center/t-perfect-center.jl")
include("../ttimes-center/t-common-center.jl")
include("../ttimes-center/t-partial-center.jl")
include("../interval-ahp.jl")

using CSV
using DataFrames
using Base.Threads

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
function run_experiments_from_csv(input_file1::String, input_file2::String, result_filename::String, result_filename2::String, result_filename3::String, result_filename4::String)
    n = 5
    
    # CSVファイルを読み込む
    df1 = CSV.read(input_file1, DataFrame)
    df2 = CSV.read(input_file2, DataFrame)

    # 全てのTrialのPCM_Identifiersを抽出
    extracted_trials = df1[:, :PCM_Identifiers]

    # 不要な部分を削除して必要な文字列のみを抽出
    cleaned_vector = String[]
    for item in extracted_trials
        # 不要な文字列を取り除く
        cleaned_item = replace(item, r"ny\[\"" => "")
        cleaned_item = replace(cleaned_item, "\"" => "")
        
        # 結果のベクターに追加
        push!(cleaned_vector, cleaned_item)
    end

    # Trialの番号を抽出
    trial_numbers = Int[]
    for item in cleaned_vector
        # "Trial-"に続く数字を抽出
        match_obj = match(r"Trial-(\d+)_PCM", item)
        if match_obj !== nothing
            trial_number = parse(Int, match_obj.captures[1])
            push!(trial_numbers, trial_number)
        end
    end

    # 重複を削除してユニークなリストを作成
    unique_trial_numbers = unique(trial_numbers)

    # PCMデータの列名を動的に生成
    pcm_columns = ["pcm_$i" for i in 1:n]

    # 各TrialごとにPCMデータと区間重要度データを格納するための辞書を作成
    trial_data = Dict()

    # CSVから読み込んだデータに対して、各Trialごとに処理を行う
    for trial_number in unique_trial_numbers
        # 各PCMごとのデータを格納するための辞書
        pcm_data = Dict()
        
        # 対応するTrialのデータを取得
        trial_df = filter(row -> row.Trial == trial_number, df2)
        
        for pcm_number in unique(trial_df.PCM)
            # 対応するPCMのデータを取得
            pcm_df = filter(row -> row.PCM == pcm_number, trial_df)
            
            # PCMデータと区間重要度データを取得
            pcm_values = Matrix{Float64}(pcm_df[:, pcm_columns])
            w_values = [interval(row.w_L, row.w_U) for row in eachrow(pcm_df)]
            
            # PCMデータ辞書に格納
            pcm_data[pcm_number] = (pcm_values, w_values)
        end
        
        # 最終的なTrialデータ辞書に格納
        trial_data[trial_number] = pcm_data
    end

    trials = 1000

    temp_results = Array{Any}(undef, trials)
    temp_results2 = Array{Any}(undef, trials)
    temp_results3 = Array{Any}(undef, trials)
    temp_results4 = Array{Any}(undef, trials)

    # trials回の実験を実行
    Threads.@threads for i in 1:trials
        A = [zeros(Float64, n, n) for _ in 1:3]
        for m = 1:3
            A[m], w = trial_data[i][m]
        end
        selected_pcms = [A[1], A[2], A[3]]
        pcm_identifiers = i

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
        CommonGround = solveCommonGroundLP2(selected_pcms)
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
            中心総和1のCommonGround = calculate_result(CommonGround, n, "common_entani_normal"),
            解の非唯一性考慮のCommonGround = calculate_result(tCommonGround2, n, "tcommon_center_1"),
            entani論文のPartialIncorporation = calculate_result(PartialIncorporation_before, n, "partial_entani"),
            中心総和1のPartialIncorporation = calculate_result(PartialIncorporation, n, "partial_center_1"),
            解の非唯一性考慮のPartialIncorporation = calculate_result(tPartialIncorporation2, n, "tpartial_center_1"),
        )

        # スレッドごとの結果を一時的に保存
        local_results_perfect = (
            Trial = i,
            entani論文のPerfectIncorporation_wᴸ = result_weights(PerfectIncorporation_before, "perfect_entani")[1],
            解の非唯一性考慮のPerfectIncorporation_wᴸ = result_weights(tPerfectIncorporation2, "tperfect_center_1")[1],
            entani論文のPerfectIncorporation_wᵁ = result_weights(PerfectIncorporation_before, "perfect_entani")[2],
            解の非唯一性考慮のPerfectIncorporation_wᵁ = result_weights(tPerfectIncorporation2, "tperfect_center_1")[2],
            entani論文のPerfectIncorporation_wᴸ_1 = result_weights_k(PerfectIncorporation_before, 1, "perfect_entani")[1],
            解の非唯一性考慮のPerfectIncorporation_wᴸ_1 = result_weights_k(tPerfectIncorporation2, 1, "tperfect_center_1")[1],
            entani論文のPerfectIncorporation_wᵁ_1 = result_weights_k(PerfectIncorporation_before, 1, "perfect_entani")[2],
            解の非唯一性考慮のPerfectIncorporation_wᵁ_1 = result_weights_k(tPerfectIncorporation2, 1, "tperfect_center_1")[2],
            entani論文のPerfectIncorporation_wᴸ_2 = result_weights_k(PerfectIncorporation_before, 2, "perfect_entani")[1],
            解の非唯一性考慮のPerfectIncorporation_wᴸ_2 = result_weights_k(tPerfectIncorporation2, 2, "tperfect_center_1")[1],
            entani論文のPerfectIncorporation_wᵁ_2 = result_weights_k(PerfectIncorporation_before, 2, "perfect_entani")[2],
            解の非唯一性考慮のPerfectIncorporation_wᵁ_2 = result_weights_k(tPerfectIncorporation2, 2, "tperfect_center_1")[2],
            entani論文のPerfectIncorporation_wᴸ_3 = result_weights_k(PerfectIncorporation_before, 3, "perfect_entani")[1],
            解の非唯一性考慮のPerfectIncorporation_wᴸ_3 = result_weights_k(tPerfectIncorporation2, 3, "tperfect_center_1")[1],
            entani論文のPerfectIncorporation_wᵁ_3 = result_weights_k(PerfectIncorporation_before, 3, "perfect_entani")[2], 
            解の非唯一性考慮のPerfectIncorporation_wᵁ_3 = result_weights_k(tPerfectIncorporation2, 3, "tperfect_center_1")[2],            
        )

        # スレッドごとの結果を一時的に保存
        local_results_common = (
            Trial = i,      
            entani論文のCommonGround_wᴸ = result_weights(CommonGround_before,  "common_entani")[1],
            解の非唯一性考慮のCommonGround_wᴸ = result_weights(tCommonGround2,  "tcommon_center_1")[1],
            entani論文のCommonGround_wᵁ = result_weights(CommonGround_before, "common_entani")[2],
            解の非唯一性考慮のCommonGround_wᵁ = result_weights(tCommonGround2, "tcommon_center_1")[2],
            entani論文のCommonGround_wᴸ_1 = result_weights_k(CommonGround_before, 1, "common_entani")[1],
            解の非唯一性考慮のCommonGround_wᴸ_1 = result_weights_k(tCommonGround2, 1, "tcommon_center_1")[1],
            entani論文のCommonGround_wᵁ_1 = result_weights_k(CommonGround_before, 1, "common_entani")[2],
            解の非唯一性考慮のCommonGround_wᵁ_1 = result_weights_k(tCommonGround2, 1, "tcommon_center_1")[2],
            entani論文のCommonGround_wᴸ_2 = result_weights_k(CommonGround_before, 2, "common_entani")[1],
            解の非唯一性考慮のCommonGround_wᴸ_2 = result_weights_k(tCommonGround2, 2, "tcommon_center_1")[1],
            entani論文のCommonGround_wᵁ_2 = result_weights_k(CommonGround_before, 2, "common_entani")[2],
            解の非唯一性考慮のCommonGround_wᵁ_2 = result_weights_k(tCommonGround2, 2, "tcommon_center_1")[2],
            entani論文のCommonGround_wᴸ_3 = result_weights_k(CommonGround_before, 3, "common_entani")[1],
            解の非唯一性考慮のCommonGround_wᴸ_3 = result_weights_k(tCommonGround2, 3, "tcommon_center_1")[1],
            entani論文のCommonGround_wᵁ_3 = result_weights_k(CommonGround_before, 3, "common_entani")[2],
            解の非唯一性考慮のCommonGround_wᵁ_3 = result_weights_k(tCommonGround2, 3, "tcommon_center_1")[2],
        )

        # スレッドごとの結果を一時的に保存
        local_results_partial = (
            Trial = i,
            entani論文のPartialIncorporation_wᴸ = result_weights(PartialIncorporation_before, "partial_entani")[1],
            解の非唯一性考慮のPartialIncorporation_wᴸ = result_weights(tPartialIncorporation2, "tpartial_center_1")[1],
            entani論文のPartialIncorporation_wᵁ = result_weights(PartialIncorporation_before, "partial_entani")[2],
            解の非唯一性考慮のPartialIncorporation_wᵁ = result_weights(tPartialIncorporation2, "tpartial_center_1")[2],
            entani論文のPartialIncorporation_wᴸ_1 = result_weights_k(PartialIncorporation_before, 1, "partial_entani")[1],
            解の非唯一性考慮のPartialIncorporation_wᴸ_1 = result_weights_k(tPartialIncorporation2, 1, "tpartial_center_1")[1],
            entani論文のPartialIncorporation_wᵁ_1 = result_weights_k(PartialIncorporation_before, 1, "partial_entani")[2],
            解の非唯一性考慮のPartialIncorporation_wᵁ_1 = result_weights_k(tPartialIncorporation2, 1, "tpartial_center_1")[2],
            entani論文のPartialIncorporation_wᴸ_2 = result_weights_k(PartialIncorporation_before, 2, "partial_entani")[1],
            解の非唯一性考慮のPartialIncorporation_wᴸ_2 = result_weights_k(tPartialIncorporation2, 2, "tpartial_center_1")[1],
            entani論文のPartialIncorporation_wᵁ_2 = result_weights_k(PartialIncorporation_before, 2, "partial_entani")[2],
            解の非唯一性考慮のPartialIncorporation_wᵁ_2 = result_weights_k(tPartialIncorporation2, 2, "tpartial_center_1")[2],
            entani論文のPartialIncorporation_wᴸ_3 = result_weights_k(PartialIncorporation_before, 3, "partial_entani")[1],
            解の非唯一性考慮のPartialIncorporation_wᴸ_3 = result_weights_k(tPartialIncorporation2, 3, "tpartial_center_1")[1],
            entani論文のPartialIncorporation_wᵁ_3 = result_weights_k(PartialIncorporation_before, 3, "partial_entani")[2],
            解の非唯一性考慮のPartialIncorporation_wᵁ_3 = result_weights_k(tPartialIncorporation2, 3, "tpartial_center_1")[2],
            entani論文のPartialIncorporation_Ŵ = getfield(PartialIncorporation_before, :w_partial_entani),
            解の非唯一性考慮のPartialIncorporation_Ŵ = getfield(tPartialIncorporation2, :Ŵ_tpartial_center_1),
        )

        temp_results[i] = local_results
        temp_results2[i] = local_results_perfect
        temp_results3[i] = local_results_common
        temp_results4[i] = local_results_partial
    end

    # 結果をDataFrameに変換
    results = DataFrame(vcat(temp_results...))
    results2 = DataFrame(vcat(temp_results2...))
    results3 = DataFrame(vcat(temp_results3...))
    results4 = DataFrame(vcat(temp_results4...))

    # 結果をCSVファイルに保存
    CSV.write(result_filename, results)
    CSV.write(result_filename2, results2)
    CSV.write(result_filename3, results3)
    CSV.write(result_filename4, results4)
end