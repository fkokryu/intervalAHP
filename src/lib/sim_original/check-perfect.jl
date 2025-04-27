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
        "entani論文のPerfectIncorporation_wᴸ", 
        "entani論文のPerfectIncorporation_wᵁ", 
        "entani論文のPerfectIncorporation_wᴸ_1", 
        "entani論文のPerfectIncorporation_wᵁ_1", 
        "entani論文のPerfectIncorporation_wᴸ_2", 
        "entani論文のPerfectIncorporation_wᵁ_2", 
        "entani論文のPerfectIncorporation_wᴸ_3", 
        "entani論文のPerfectIncorporation_wᵁ_3"
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

function convert_to_vectors_and_matrices(df::DataFrame)
    # ベクトルと行列を初期化
    wᴸ_perfect = Vector{Float64}[]
    wᵁ_perfect = Vector{Float64}[]
    ŵᴸ_perfect = Matrix{Float64}[]
    ŵᵁ_perfect = Matrix{Float64}[]

    for row in eachrow(df)
        # 各列から文字列を抽出し、数値の配列に変換
        wᴸ = parse.(Float64, split(row.entani論文のPerfectIncorporation_wᴸ[2:end-1], ", "))
        wᵁ = parse.(Float64, split(row.entani論文のPerfectIncorporation_wᵁ[2:end-1], ", "))
        ŵᴸ_1 = parse.(Float64, split(row.entani論文のPerfectIncorporation_wᴸ_1[2:end-1], ", "))
        ŵᵁ_1 = parse.(Float64, split(row.entani論文のPerfectIncorporation_wᵁ_1[2:end-1], ", "))
        ŵᴸ_2 = parse.(Float64, split(row.entani論文のPerfectIncorporation_wᴸ_2[2:end-1], ", "))
        ŵᵁ_2 = parse.(Float64, split(row.entani論文のPerfectIncorporation_wᵁ_2[2:end-1], ", "))
        ŵᴸ_3 = parse.(Float64, split(row.entani論文のPerfectIncorporation_wᴸ_3[2:end-1], ", "))
        ŵᵁ_3 = parse.(Float64, split(row.entani論文のPerfectIncorporation_wᵁ_3[2:end-1], ", "))

        # ベクトルに追加
        push!(wᴸ_perfect, wᴸ)
        push!(wᵁ_perfect, wᵁ)

        # 行列に変換して追加
        push!(ŵᴸ_perfect, hcat(ŵᴸ_1, ŵᴸ_2, ŵᴸ_3))
        push!(ŵᵁ_perfect, hcat(ŵᵁ_1, ŵᵁ_2, ŵᵁ_3))
    end

    return wᴸ_perfect, wᵁ_perfect, ŵᴸ_perfect, ŵᵁ_perfect
end

function is_feasible_solution(wᴸ, wᵁ, ŵᴸ, ŵᵁ, s, matrices)
    ε = 1e-8 # << 1
    l = length(matrices) # 人数
    m,n = size(matrices[1])

    if !all(Aₖ -> isCrispPCM(Aₖ), matrices)
        throw(ArgumentError("Aₖ is not a crisp PCM"))
    end
    
    if !all(Aₖ -> size(Aₖ) == (n, n), matrices)
        throw(ArgumentError("Some matrices have different size"))
    end

    ḋ_perfect_model = map(Aₖ -> solveCrispAHPLP(Aₖ).optimalValue_center_1, matrices)

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    # 定義された変数をモデルに追加し、与えられた解を変数に代入
    @variable(model, wᴸ_perfect_model[i=1:n] >= ε)
    @variable(model, wᵁ_perfect_model[i=1:n] >= ε)
    @variable(model, ŵᴸ_perfect_model[k=1:l, i=1:n] >= ε)
    @variable(model, ŵᵁ_perfect_model[k=1:l, i=1:n] >= ε)
    @variable(model, s_perfect_model ≥ ε)

    for i in 1:n
        set_start_value(wᴸ_perfect_model[i], wᴸ[i])
        fix(wᴸ_perfect_model[i], wᴸ[i], force=true) # force=trueを追加
        set_start_value(wᵁ_perfect_model[i], wᵁ[i])
        fix(wᵁ_perfect_model[i], wᵁ[i], force=true) # force=trueを追加
        for k in 1:l
            set_start_value(ŵᴸ_perfect_model[k, i], ŵᴸ[i, k])
            fix(ŵᴸ_perfect_model[k, i], ŵᴸ[i, k], force=true) # force=trueを追加
            set_start_value(ŵᵁ_perfect_model[k, i], ŵᵁ[i, k])
            fix(ŵᵁ_perfect_model[k, i], ŵᵁ[i, k], force=true) # force=trueを追加
        end
    end
    
    set_start_value(s_perfect_model, s)
    fix(s_perfect_model, s, force=true) # force=trueを追加

    # 最適化を実行する前に各変数の初期値を保存
    initial_wᴸ = copy(wᴸ) # 深いコピーを使って元のデータを保持
    initial_wᵁ = copy(wᵁ)
    initial_ŵᴸ = transpose(ŵᴸ)
    initial_ŵᵁ = transpose(ŵᵁ)
    initial_s = s

    println("元の最適値:", sum(initial_wᵁ)-sum(initial_wᴸ))

    # ここに制約を追加
    for k = 1:l
        ŵₖᴸ_perfect_model = ŵᴸ_perfect_model[k,:]; ŵₖᵁ_perfect_model = ŵᵁ_perfect_model[k,:]

        Aₖ = matrices[k]

        # ∑(ŵₖᵢᵁ_perfect_model - ŵₖᵢᴸ_perfect_model) ≤ sₖḋ_perfect_modelₖ
        @constraint(model, sum(ŵₖᵁ_perfect_model) - sum(ŵₖᴸ_perfect_model) ≤ ((sum(ŵₖᴸ_perfect_model) + sum(ŵₖᵁ_perfect_model)) / 2 * (ḋ_perfect_model[k]) + ε) )

        for i = 1:n-1
            ŵₖᵢᴸ_perfect_model = ŵₖᴸ_perfect_model[i]; ŵₖᵢᵁ_perfect_model = ŵₖᵁ_perfect_model[i]

            for j = i+1:n
                aₖᵢⱼ = Aₖ[i,j]
                ŵₖⱼᴸ_perfect_model = ŵₖᴸ_perfect_model[j]; ŵₖⱼᵁ_perfect_model = ŵₖᵁ_perfect_model[j]
                
                @constraint(model, ŵₖᵢᴸ_perfect_model ≤ aₖᵢⱼ * ŵₖⱼᵁ_perfect_model)
                @constraint(model, aₖᵢⱼ * ŵₖⱼᴸ_perfect_model ≤ ŵₖᵢᵁ_perfect_model)
            end
        end

        for i = 1:n
            ŵₖᵢᴸ_perfect_model = ŵₖᴸ_perfect_model[i]; ŵₖᵢᵁ_perfect_model = ŵₖᵁ_perfect_model[i]

            # 正規性条件
            ∑ŵₖⱼᴸ_perfect_model = sum(map(j -> ŵₖᴸ_perfect_model[j], filter(j -> i != j, 1:n)))
            @constraint(model, ∑ŵₖⱼᴸ_perfect_model + ŵₖᵢᵁ_perfect_model ≤ s_perfect_model)
            ∑ŵₖⱼᵁ_perfect_model = sum(map(j -> ŵₖᵁ_perfect_model[j], filter(j -> i != j, 1:n)))
            @constraint(model, ∑ŵₖⱼᵁ_perfect_model + ŵₖᵢᴸ_perfect_model ≥ s_perfect_model)

            wᵢᴸ_perfect_model = wᴸ_perfect_model[i]; wᵢᵁ_perfect_model = wᵁ_perfect_model[i] 
            @constraint(model, ŵₖᵢᴸ_perfect_model ≥ wᵢᴸ_perfect_model)
            @constraint(model, ŵₖᵢᵁ_perfect_model ≥ ŵₖᵢᴸ_perfect_model)
            @constraint(model, wᵢᵁ_perfect_model ≥ ŵₖᵢᵁ_perfect_model)
        end
    end

    @constraint(model, sum(wᵁ_perfect_model) + sum(wᴸ_perfect_model) == 2)

    @objective(model, Min, sum(wᵁ_perfect_model) - sum(wᴸ_perfect_model))

    # モデルの実行可能性チェック
    optimize!(model)

    # 最適化後の変数の値を取得
    optimized_wᴸ_perfect_model = [value(wᴸ_perfect_model[i]) for i in 1:n]
    optimized_wᵁ_perfect_model = [value(wᵁ_perfect_model[i]) for i in 1:n]
    optimized_ŵᴸ_perfect_model = [value(ŵᴸ_perfect_model[k, i]) for k in 1:l, i in 1:n]
    optimized_ŵᵁ_perfect_model = [value(ŵᵁ_perfect_model[k, i]) for k in 1:l, i in 1:n]
    optimized_s_perfect_model = value(s_perfect_model)

    # 変更を計算
    Δwᴸ_perfect_model = optimized_wᴸ_perfect_model - initial_wᴸ
    Δwᵁ_perfect_model = optimized_wᵁ_perfect_model - initial_wᵁ
    Δŵᴸ_perfect_model = optimized_ŵᴸ_perfect_model - initial_ŵᴸ
    Δŵᵁ_perfect_model = optimized_ŵᵁ_perfect_model - initial_ŵᵁ
    Δs_perfect_model = optimized_s_perfect_model - initial_s

    # 変更を出力
    println("変更後の最適値:", sum(optimized_wᵁ_perfect_model)-sum(optimized_wᴸ_perfect_model))
    println("Δwᴸ_perfect_model: ", sum(Δwᴸ_perfect_model))
    println("Δwᵁ_perfect_model: ", sum(Δwᵁ_perfect_model))
    println("Δŵᴸ_perfect_model: ", Δŵᴸ_perfect_model)
    println("Δŵᵁ_perfect_model: ", Δŵᵁ_perfect_model)
    println("Δs_perfect_model: ", Δs_perfect_model)

    return termination_status(model) != MOI.INFEASIBLE
end


function calculate_t(wᴸ_perfect::Vector{Vector{Float64}}, wᵁ_perfect::Vector{Vector{Float64}})
    n = length(wᴸ_perfect)
    t = Vector{Float64}(undef, n)

    for i in 1:n
        wᴸ = wᴸ_perfect[i]
        wᵁ = wᵁ_perfect[i]

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
function load_and_process_csv_perfect(input_file1::String, input_file2::String, input_file3::String , output_file::String)
    # CSVファイルを読み込む
    df1 = CSV.read(input_file1, DataFrame)
    df2 = CSV.read(input_file3, DataFrame)

    # `式(1)の最適値が式(10)の最適値のt_L倍より大きい` 列から数値を取得
    selected_values = df1[.!(ismissing.(df1[:, :"式(1)の最適値が式(10)の最適値のt_L倍より大きい"])) .& 
                      (df1[:, :"式(1)の最適値が式(10)の最適値のt_L倍より大きい"] .!= "FALSE"), :"式(1)の最適値が式(10)の最適値のt_L倍より大きい"]
    
    # String7からStringへの変換
    unique_trial_numbers = String.(unique(selected_values))

    relevant_data = extract_relevant_data(input_file2, unique_trial_numbers)

    wᴸ_perfect, wᵁ_perfect, ŵᴸ_perfect, ŵᵁ_perfect = convert_to_vectors_and_matrices(relevant_data)

    t = calculate_t(wᴸ_perfect, wᵁ_perfect)

    A = extract_pcm_data(df2, unique_trial_numbers)

    results_df = DataFrame(Trial = String[], Feasibility = Bool[])

    for i in 1:length(unique_trial_numbers)
        pcm1, w1 = A[unique_trial_numbers[i]][1]
        pcm2, w2 = A[unique_trial_numbers[i]][2]
        pcm3, w3 = A[unique_trial_numbers[i]][3]

        # 実行可能性をチェック
        feasibility = is_feasible_solution(t[i]*wᴸ_perfect[i], t[i]*wᵁ_perfect[i], t[i]*ŵᴸ_perfect[i], t[i]*ŵᵁ_perfect[i], t[i], [pcm1,pcm2,pcm3])

        # 結果をDataFrameに追加
        push!(results_df, (Trial = unique_trial_numbers[i], Feasibility = feasibility))
    end

    # 結果をCSVファイルに書き出し
    CSV.write(output_file, results_df)
end