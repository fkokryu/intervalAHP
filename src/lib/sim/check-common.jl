using CSV
using DataFrames
using IntervalArithmetic
using JuMP
import HiGHS
import MathOptInterface as MOI  # これを追加

include("../crisp-pcm.jl")
include("../nearly-equal.jl")
include("../interval-ahp.jl")
include("../ttimes/optimal-value.jl")

function extract_relevant_data(input_file::String, trial_numbers::Vector{String})
    # CSVファイルを読み込む
    df = CSV.read(input_file, DataFrame)

    # 列名のリスト
    column_names = [
        "Trial",
        "entani論文のCommonGround_wᴸ", 
        "entani論文のCommonGround_wᵁ", 
        "entani論文のCommonGround_wᴸ_1", 
        "entani論文のCommonGround_wᵁ_1", 
        "entani論文のCommonGround_wᴸ_2", 
        "entani論文のCommonGround_wᵁ_2", 
        "entani論文のCommonGround_wᴸ_3", 
        "entani論文のCommonGround_wᵁ_3"
    ]

    # 対象のトライアルのデータを抽出
    filtered_data = filter(row -> string(row.Trial) in trial_numbers, df)

    # 必要な列だけを選択
    relevant_data = select(filtered_data, column_names)

    return relevant_data
end

# 特定のトライアルに対してチェックを行う関数
function check_diff_for_trial(df, trial_number)
    diffs = []
    # 特定のトライアルのデータを選択
    trial_data = filter(row -> row.Trial == trial_number, df)
    
    # トライアルデータが空の場合（該当するトライアル番号がない場合）
    if isempty(trial_data)
        return "Trial not found"
    end
    
    # DM1, DM2, DM3について差をチェック
    for dm in 1:3
        最適値 = trial_data[!, Symbol("DM$(dm)の式1の最適値")][1]
        積 = trial_data[!, Symbol("DM$(dm)の積")][1]
        if abs(最適値 - 積) > 1e-6
            push!(diffs, dm)
        end
    end
    
    return diffs
end

function convert_to_vectors_and_matrices(df::DataFrame)
    # ベクトルと行列を初期化
    wᴸ_common = Vector{Float64}[]
    wᵁ_common = Vector{Float64}[]
    ŵᴸ_common = Matrix{Float64}[]
    ŵᵁ_common = Matrix{Float64}[]

    for row in eachrow(df)
        # 各列から文字列を抽出し、数値の配列に変換
        wᴸ = parse.(Float64, split(row.entani論文のCommonGround_wᴸ[2:end-1], ", "))
        wᵁ = parse.(Float64, split(row.entani論文のCommonGround_wᵁ[2:end-1], ", "))
        ŵᴸ_1 = parse.(Float64, split(row.entani論文のCommonGround_wᴸ_1[2:end-1], ", "))
        ŵᵁ_1 = parse.(Float64, split(row.entani論文のCommonGround_wᵁ_1[2:end-1], ", "))
        ŵᴸ_2 = parse.(Float64, split(row.entani論文のCommonGround_wᴸ_2[2:end-1], ", "))
        ŵᵁ_2 = parse.(Float64, split(row.entani論文のCommonGround_wᵁ_2[2:end-1], ", "))
        ŵᴸ_3 = parse.(Float64, split(row.entani論文のCommonGround_wᴸ_3[2:end-1], ", "))
        ŵᵁ_3 = parse.(Float64, split(row.entani論文のCommonGround_wᵁ_3[2:end-1], ", "))

        # ベクトルに追加
        push!(wᴸ_common, wᴸ)
        push!(wᵁ_common, wᵁ)

        # 行列に変換して追加
        push!(ŵᴸ_common, hcat(ŵᴸ_1, ŵᴸ_2, ŵᴸ_3))
        push!(ŵᵁ_common, hcat(ŵᵁ_1, ŵᵁ_2, ŵᵁ_3))
    end

    return wᴸ_common, wᵁ_common, ŵᴸ_common, ŵᵁ_common
end

function is_feasible_solution(diff, matrices)
    ε = 1e-8 # << 1
    l = length(matrices) # 人数
    m,n = size(matrices[1])

    if !all(Aₖ -> isCrispPCM(Aₖ), matrices)
        throw(ArgumentError("Aₖ is not a crisp PCM"))
    end
    
    if !all(Aₖ -> size(Aₖ) == (n, n), matrices)
        throw(ArgumentError("Some matrices have different size"))
    end

    ḋ_common_model = map(Aₖ -> solveCrispAHPLP(Aₖ).optimalValue_center_1, matrices)

    #数値実験より「式(1)の最適値」÷「式(1)の最適解の中心の総和」＜「式(10)の最適値」、つまり「式(1)の最適値」÷「式(1)の最適解の中心の総和」が正しい解
    w = map(Aₖ -> solveIntervalAHPLP(Aₖ), matrices)
    d_common_model = map(result -> result.optimalValue, w)
    wᴸ_individual = map(result -> result.wᴸ, w)
    wᵁ_individual = map(result -> result.wᵁ, w)

    for l in diff
        ḋ_common_model[l] = d_common_model[l] / ((sum(wᴸ_individual[l])+sum(wᵁ_individual[l]))/2)
    end

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    # 定義された変数をモデルに追加
    @variable(model, wᴸ_common_model[i=1:n] ≥ ε)
    @variable(model, wᵁ_common_model[i=1:n] ≥ ε)
    @variable(model, ŵᴸ_common_model[k=1:l,i=1:n] ≥ ε)
    @variable(model, ŵᵁ_common_model[k=1:l,i=1:n] ≥ ε)
    @variable(model, s_common_model ≥ ε)

    # ここに制約を追加
    for k = 1:l
        ŵₖᴸ_common_model = ŵᴸ_common_model[k,:]; ŵₖᵁ_common_model = ŵᵁ_common_model[k,:]

        Aₖ = matrices[k]

        # ∑(ŵₖᵢᵁ_common_model - ŵₖᵢᴸ_common_model) ≤ sₖḋ_common_modelₖ
        @constraint(model, sum(ŵₖᵁ_common_model) - sum(ŵₖᴸ_common_model) ≤ ((sum(ŵₖᴸ_common_model) + sum(ŵₖᵁ_common_model)) / 2 * (ḋ_common_model[k] + ε)))

        for i = 1:n-1
            ŵₖᵢᴸ_common_model = ŵₖᴸ_common_model[i]; ŵₖᵢᵁ_common_model = ŵₖᵁ_common_model[i]

            for j = i+1:n
                aₖᵢⱼ = Aₖ[i,j]
                ŵₖⱼᴸ_common_model = ŵₖᴸ_common_model[j]; ŵₖⱼᵁ_common_model = ŵₖᵁ_common_model[j]
                
                @constraint(model, ŵₖᵢᴸ_common_model ≤ aₖᵢⱼ * ŵₖⱼᵁ_common_model)
                @constraint(model, aₖᵢⱼ * ŵₖⱼᴸ_common_model ≤ ŵₖᵢᵁ_common_model)
            end
        end

        for i = 1:n
            ŵₖᵢᴸ_common_model = ŵₖᴸ_common_model[i]; ŵₖᵢᵁ_common_model = ŵₖᵁ_common_model[i]

            # 正規性条件
            ∑ŵₖⱼᴸ_common_model = sum(map(j -> ŵₖᴸ_common_model[j], filter(j -> i != j, 1:n)))
            @constraint(model, ∑ŵₖⱼᴸ_common_model + ŵₖᵢᵁ_common_model ≤ s_common_model)
            ∑ŵₖⱼᵁ_common_model = sum(map(j -> ŵₖᵁ_common_model[j], filter(j -> i != j, 1:n)))
            @constraint(model, ∑ŵₖⱼᵁ_common_model + ŵₖᵢᴸ_common_model ≥ s_common_model)

            wᵢᴸ_common_model = wᴸ_common_model[i]; wᵢᵁ_common_model = wᵁ_common_model[i]
            @constraint(model, wᵢᴸ_common_model ≥ ŵₖᵢᴸ_common_model)
            @constraint(model, wᵢᵁ_common_model ≥ wᵢᴸ_common_model)
            @constraint(model, ŵₖᵢᵁ_common_model ≥ wᵢᵁ_common_model)
        end
    end

    @constraint(model, sum(wᵁ_common_model) + sum(wᴸ_common_model) == 2)

    for i = 1:n
        wᵢᴸ_common_model = wᴸ_common_model[i]; wᵢᵁ_common_model = wᵁ_common_model[i] 
        # kなしの正規性条件
        ∑wⱼᴸ_common_model = sum(map(j -> wᴸ_common_model[j], filter(j -> i != j, 1:n)))
        @constraint(model, ∑wⱼᴸ_common_model + wᵢᵁ_common_model ≤ 1)
        ∑wⱼᵁ_common_model = sum(map(j -> wᵁ_common_model[j], filter(j -> i != j, 1:n)))
        @constraint(model, ∑wⱼᵁ_common_model + wᵢᴸ_common_model ≥ 1)
    end

    # 目的関数 ∑(wᵢᵁ_common_model - wᵢᴸ_common_model)
    @objective(model, Max, sum(wᵁ_common_model) - sum(wᴸ_common_model))

    # モデルの実行と実行可能性の確認
    optimize!(model)

    return (n - 1) * sum(log.(value.(wᵁ_common_model)) - log.(value.(wᴸ_common_model)))
end

function calculate_t(wᴸ_common::Vector{Vector{Float64}}, wᵁ_common::Vector{Vector{Float64}})
    n = length(wᴸ_common)
    t = Vector{Float64}(undef, n)

    for i in 1:n
        wᴸ = wᴸ_common[i]
        wᵁ = wᵁ_common[i]

        # t[i] の計算
        t[i] = 1 / sum((wᴸ + wᵁ) / 2)
    end

    return t
end

function extract_pcm_data(df::DataFrame, unique_trial_numbers::Vector{String})
    # PCMデータの列名を動的に生成
    n = 5
    pcm_columns = ["pcm_$i" for i in 1:n]

    # 各TrialごとにPCMデータと区間重要度データを格納するための辞書を作成
    trial_data = Dict()

    # CSVから読み込んだデータに対して、各Trialごとに処理を行う
    for trial_number in unique_trial_numbers
        # 各PCMごとのデータを格納するための辞書
        pcm_data = Dict()
        
        # 対応するTrialのデータを取得
        trial_df = filter(row -> string(row.Trial) == trial_number, df)
        
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

    return trial_data
end

function get_t_L_value(trial_number::Int, pcm_number::Int, df::DataFrame)
    row = df[(df.Trial .== trial_number) .& (df.PCM .== pcm_number), :]
    return isempty(row) ? missing : row[1, :"式10の解に対するt_L"]
end

# CSVファイルからデータを読み込む関数
function load_and_process_csv_common(input_file1::String, input_file2::String, input_file3::String, input_file4::String, output_file::String)
    # CSVファイルを読み込む
    df1 = CSV.read(input_file1, DataFrame)
    df2 = CSV.read(input_file3, DataFrame)

    # `式(1)の最適値が式(10)の最適値のt_L倍より大きい` 列から数値を取得
    selected_values = df1[.!(ismissing.(df1[:, :"式(1)の最適値が式(10)の最適値のt_L倍より大きい"])) .& 
                      (df1[:, :"式(1)の最適値が式(10)の最適値のt_L倍より大きい"] .!= "FALSE"), :"式(1)の最適値が式(10)の最適値のt_L倍より大きい"]

    selected_columns_df = df1[:, ["Trial", "PCM", "式10の解に対するt_L"]]
    
    # String7からStringへの変換
    unique_trial_numbers = String.(unique(selected_values))

    relevant_data = extract_relevant_data(input_file2, unique_trial_numbers)

    wᴸ_common, wᵁ_common, ŵᴸ_common, ŵᵁ_common = convert_to_vectors_and_matrices(relevant_data)

    #t = calculate_t(wᴸ_common, wᵁ_common)

    A = extract_pcm_data(df2, unique_trial_numbers)

    results_df = DataFrame(Trial = String[], Feasibility = Float64[])

    df =  CSV.read(input_file4, DataFrame)

    for i in 1:length(unique_trial_numbers)
        trial_number_int = parse(Int, unique_trial_numbers[i])  # String を Int に変換

        pcm1, w1 = A[unique_trial_numbers[i]][1]
        pcm2, w2 = A[unique_trial_numbers[i]][2]
        pcm3, w3 = A[unique_trial_numbers[i]][3]

        t_L = [get_t_L_value(trial_number_int, 1, selected_columns_df), get_t_L_value(trial_number_int, 2, selected_columns_df), get_t_L_value(trial_number_int, 3, selected_columns_df)]

        # 実行可能性をチェック
        feasibility = is_feasible_solution(check_diff_for_trial(df, trial_number_int), [pcm1,pcm2,pcm3])

        # 結果をDataFrameに追加
        push!(results_df, (Trial = unique_trial_numbers[i], Feasibility = feasibility))
    end

    # 結果をCSVファイルに書き出し
    CSV.write(output_file, results_df)
end