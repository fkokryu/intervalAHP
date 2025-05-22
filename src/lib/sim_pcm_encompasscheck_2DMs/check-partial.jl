using CSV
using DataFrames
using IntervalArithmetic
using JuMP
import HiGHS
import MathOptInterface as MOI  # これを追加

include("../crisp-pcm.jl")
include("../nearly-equal.jl")
include("../ttimes/optimal-value.jl")

function extract_relevant_data(input_file::String, trial_numbers::Vector{String})
    # CSVファイルを読み込む
    df = CSV.read(input_file, DataFrame)

    # 列名のリスト
    column_names = [
        "Trial",
        "entani論文のPartialIncorporation_wᴸ", 
        "entani論文のPartialIncorporation_wᵁ", 
        "entani論文のPartialIncorporation_wᴸ_1", 
        "entani論文のPartialIncorporation_wᵁ_1", 
        "entani論文のPartialIncorporation_wᴸ_2", 
        "entani論文のPartialIncorporation_wᵁ_2", 
        "entani論文のPartialIncorporation_wᴸ_3", 
        "entani論文のPartialIncorporation_wᵁ_3",
        "entani論文のPartialIncorporation_Ŵ"
    ]

    # 対象のトライアルのデータを抽出
    filtered_data = filter(row -> string(row.Trial) in trial_numbers, df)

    # 必要な列だけを選択
    relevant_data = select(filtered_data, column_names)

    return relevant_data
end

# 使用例
# unique_trial_numbers = load_and_process_csv("input_file1.csv", "input_file2.csv")
# relevant_data = extract_relevant_data("weights_5_3_0.65.csv", unique_trial_numbers)

function split_and_parse_w(row)
    # 文字列から括弧を削除
    cleaned_string = replace(row.entani論文のPartialIncorporation_Ŵ, '[' => "", ']' => "")

    # 文字列を分割し、Float64に変換
    w_data = parse.(Float64, split(cleaned_string, ", "))

    # データを3つのベクトルに分割
    Ŵ_1 = w_data[1:5]  # 最初の5要素
    Ŵ_2 = w_data[6:10] # 次の5要素
    Ŵ_3 = w_data[11:15] # 最後の5要素

    return Ŵ_1, Ŵ_2, Ŵ_3
end

function convert_to_vectors_and_matrices(df::DataFrame)
    # ベクトルと行列を初期化
    wᴸ_partial = Vector{Float64}[]
    wᵁ_partial = Vector{Float64}[]
    ŵᴸ_partial = Matrix{Float64}[]
    ŵᵁ_partial = Matrix{Float64}[]
    Ŵ_partial = Matrix{Float64}[]

    for row in eachrow(df)
        # 各列から文字列を抽出し、数値の配列に変換
        wᴸ = parse.(Float64, split(row.entani論文のPartialIncorporation_wᴸ[2:end-1], ", "))
        wᵁ = parse.(Float64, split(row.entani論文のPartialIncorporation_wᵁ[2:end-1], ", "))
        ŵᴸ_1 = parse.(Float64, split(row.entani論文のPartialIncorporation_wᴸ_1[2:end-1], ", "))
        ŵᵁ_1 = parse.(Float64, split(row.entani論文のPartialIncorporation_wᵁ_1[2:end-1], ", "))
        ŵᴸ_2 = parse.(Float64, split(row.entani論文のPartialIncorporation_wᴸ_2[2:end-1], ", "))
        ŵᵁ_2 = parse.(Float64, split(row.entani論文のPartialIncorporation_wᵁ_2[2:end-1], ", "))
        ŵᴸ_3 = parse.(Float64, split(row.entani論文のPartialIncorporation_wᴸ_3[2:end-1], ", "))
        ŵᵁ_3 = parse.(Float64, split(row.entani論文のPartialIncorporation_wᵁ_3[2:end-1], ", "))

        Ŵ_1, Ŵ_2, Ŵ_3 = split_and_parse_w(row)

        # ベクトルに追加
        push!(wᴸ_partial, wᴸ)
        push!(wᵁ_partial, wᵁ)

        # 行列に変換して追加
        push!(ŵᴸ_partial, hcat(ŵᴸ_1, ŵᴸ_2, ŵᴸ_3))
        push!(ŵᵁ_partial, hcat(ŵᵁ_1, ŵᵁ_2, ŵᵁ_3))

        push!(Ŵ_partial, hcat(Ŵ_1, Ŵ_2, Ŵ_3))
    end

    return wᴸ_partial, wᵁ_partial, ŵᴸ_partial, ŵᵁ_partial, Ŵ_partial
end

function is_feasible_solution(wᴸ, wᵁ, ŵᴸ, ŵᵁ, Ŵ, s, matrices)
    ε = 1e-8 # << 1
    l = length(matrices) # 人数
    m,n = size(matrices[1])

    if !all(Aₖ -> isCrispPCM(Aₖ), matrices)
        throw(ArgumentError("Aₖ is not a crisp PCM"))
    end
    
    if !all(Aₖ -> size(Aₖ) == (n, n), matrices)
        throw(ArgumentError("Some matrices have different size"))
    end

    ḋ_partial_model = map(Aₖ -> solveCrispAHPLP(Aₖ).optimalValue_center_1, matrices)

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    # 定義された変数をモデルに追加
    @variable(model, wᴸ_partial_model[i=1:n] >= ε)
    @variable(model, wᵁ_partial_model[i=1:n] >= ε)
    @variable(model, ŵᴸ_partial_model[k=1:l, i=1:n] >= ε)
    @variable(model, ŵᵁ_partial_model[k=1:l, i=1:n] >= ε)
    @variable(model, Ŵ_partial_model[k=1:l,i=1:n] ≥ ε)
    @variable(model, s_partial_model ≥ ε)

    # 与えられた解をモデルの変数に代入
    for i in 1:n
        set_start_value(wᴸ_partial_model[i], wᴸ[i])
        set_start_value(wᵁ_partial_model[i], wᵁ[i])
        for k in 1:l
            set_start_value(ŵᴸ_partial_model[k, i], ŵᴸ[i, k])
            set_start_value(ŵᵁ_partial_model[k, i], ŵᵁ[i, k])
            set_start_value(Ŵ_partial_model[k, i], Ŵ[i, k])
        end
    end

    set_start_value(s_partial_model, s)

    # ここに制約を追加
    for k = 1:l
        ŵₖᴸ_partial_model = ŵᴸ_partial_model[k,:]; ŵₖᵁ_partial_model = ŵᵁ_partial_model[k,:]
        Ŵ_partial_modelₖ = Ŵ_partial_model[k,:]

        Aₖ = matrices[k]

        # ∑(ŵₖᵢᵁ_partial_model - ŵₖᵢᴸ_partial_model) ≤ sₖḋ_partial_modelₖ
        @constraint(model, sum(ŵₖᵁ_partial_model) - sum(ŵₖᴸ_partial_model) ≤ ((sum(ŵₖᴸ_partial_model) + sum(ŵₖᵁ_partial_model)) / 2 * (ḋ_partial_model[k] + ε)))

        for i = 1:n-1
            ŵₖᵢᴸ_partial_model = ŵₖᴸ_partial_model[i]; ŵₖᵢᵁ_partial_model = ŵₖᵁ_partial_model[i]

            for j = i+1:n
                aₖᵢⱼ = Aₖ[i,j]
                ŵₖⱼᴸ_partial_model = ŵₖᴸ_partial_model[j]; ŵₖⱼᵁ_partial_model = ŵₖᵁ_partial_model[j]

                @constraint(model, ŵₖᵢᴸ_partial_model ≤ aₖᵢⱼ * ŵₖⱼᵁ_partial_model)
                @constraint(model, aₖᵢⱼ * ŵₖⱼᴸ_partial_model ≤ ŵₖᵢᵁ_partial_model)
            end
        end

        @constraint(model, sum(wᵁ_partial_model) + sum(wᴸ_partial_model) == 2)

        # 正規性条件
        @constraint(model, sum(Ŵ_partial_modelₖ) == s_partial_model)

        for i = 1:n
            ŵₖᵢᴸ_partial_model = ŵₖᴸ_partial_model[i]; ŵₖᵢᵁ_partial_model = ŵₖᵁ_partial_model[i]
            wᵢᴸ_partial_model = wᴸ_partial_model[i]; wᵢᵁ_partial_model = wᵁ_partial_model[i]
            Ŵ_partial_modelₖᵢ = Ŵ_partial_modelₖ[i]

            # 正規性条件
            ∑ŵₖⱼᴸ_partial_model = sum(map(j -> ŵₖᴸ_partial_model[j], filter(j -> i != j, 1:n)))
            @constraint(model, ∑ŵₖⱼᴸ_partial_model + ŵₖᵢᵁ_partial_model ≤ s_partial_model)
            ∑ŵₖⱼᵁ_partial_model = sum(map(j -> ŵₖᵁ_partial_model[j], filter(j -> i != j, 1:n)))
            @constraint(model, ∑ŵₖⱼᵁ_partial_model + ŵₖᵢᴸ_partial_model ≥ s_partial_model)

            @constraint(model, Ŵ_partial_modelₖᵢ ≥ wᵢᴸ_partial_model)
            @constraint(model, ŵₖᵢᵁ_partial_model ≥ Ŵ_partial_modelₖᵢ)

            @constraint(model, Ŵ_partial_modelₖᵢ ≥ ŵₖᵢᴸ_partial_model)
            @constraint(model, wᵢᵁ_partial_model ≥ Ŵ_partial_modelₖᵢ)
        end
    end

    # モデルの実行可能性チェック
    optimize!(model)
    return termination_status(model) != MOI.INFEASIBLE
end


function calculate_t(wᴸ_partial::Vector{Vector{Float64}}, wᵁ_partial::Vector{Vector{Float64}})
    n = length(wᴸ_partial)
    t = Vector{Float64}(undef, n)

    for i in 1:n
        wᴸ = wᴸ_partial[i]
        wᵁ = wᵁ_partial[i]

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

# CSVファイルからデータを読み込む関数
function load_and_process_csv_partial(input_file1::String, input_file2::String, input_file3::String , output_file::String)
    # CSVファイルを読み込む
    df1 = CSV.read(input_file1, DataFrame)
    df2 = CSV.read(input_file3, DataFrame)

    # `式(1)の最適値が式(10)の最適値のt_L倍より大きい` 列から数値を取得
    selected_values = df1[.!(ismissing.(df1[:, :"式(1)の最適値が式(10)の最適値のt_L倍より大きい"])) .& 
                      (df1[:, :"式(1)の最適値が式(10)の最適値のt_L倍より大きい"] .!= "FALSE"), :"式(1)の最適値が式(10)の最適値のt_L倍より大きい"]
    
    # String7からStringへの変換
    unique_trial_numbers = String.(unique(selected_values))

    relevant_data = extract_relevant_data(input_file2, unique_trial_numbers)

    wᴸ_partial, wᵁ_partial, ŵᴸ_partial, ŵᵁ_partial, Ŵ_partial = convert_to_vectors_and_matrices(relevant_data)

    t = calculate_t(wᴸ_partial, wᵁ_partial)

    A = extract_pcm_data(df2, unique_trial_numbers)

    results_df = DataFrame(Trial = String[], Feasibility = Bool[])

    for i in 1:length(unique_trial_numbers)
        pcm1, w1 = A[unique_trial_numbers[i]][1]
        pcm2, w2 = A[unique_trial_numbers[i]][2]
        pcm3, w3 = A[unique_trial_numbers[i]][3]

        # 実行可能性をチェック
        feasibility = is_feasible_solution(wᴸ_partial[i], wᵁ_partial[i], ŵᴸ_partial[i], ŵᵁ_partial[i], Ŵ_partial[i], t[i], [pcm1,pcm2,pcm3])

        # 結果をDataFrameに追加
        push!(results_df, (Trial = unique_trial_numbers[i], Feasibility = feasibility))
    end

    # 結果をCSVファイルに書き出し
    CSV.write(output_file, results_df)
end

