using CSV
using DataFrames
using IntervalArithmetic
using JuMP
import HiGHS
import MathOptInterface as MOI
using Plots
using LinearAlgebra

# 既存のファイルをインクルード（必要に応じて）
# include("../crisp-pcm.jl")
# include("../nearly-equal.jl")
# include("../ttimes/optimal-value.jl")

function extract_interval_weights(input_file::String, trial_numbers::Vector{Int})
    # CSVファイルを読み込む
    df = CSV.read(input_file, DataFrame)
    
    # 対象のトライアルのデータを抽出
    filtered_data = filter(row -> row.Trial in trial_numbers, df)
    
    return filtered_data
end

function parse_weight_vector(weight_str::String)
    # 文字列が "E" の場合や欠損値の場合は空のベクトルを返す
    if ismissing(weight_str) || weight_str == "E"
        return Float64[]
    end
    
    # 文字列から括弧を取り除き、カンマで分割して数値に変換
    return parse.(Float64, split(weight_str[2:end-1], ", "))
end

function convert_weights_to_matrices(df::DataFrame)
    n_rows = nrow(df)
    
    # 各手法・各DMの重みを格納する配列
    entani_wᴸ = Vector{Vector{Float64}}(undef, n_rows)
    entani_wᵁ = Vector{Vector{Float64}}(undef, n_rows)
    nonunique_wᴸ = Vector{Vector{Float64}}(undef, n_rows)
    nonunique_wᵁ = Vector{Vector{Float64}}(undef, n_rows)
    
    # 各DMの重みを格納する配列
    entani_wᴸ_dms = Vector{Matrix{Float64}}(undef, n_rows)
    entani_wᵁ_dms = Vector{Matrix{Float64}}(undef, n_rows)
    nonunique_wᴸ_dms = Vector{Matrix{Float64}}(undef, n_rows)
    nonunique_wᵁ_dms = Vector{Matrix{Float64}}(undef, n_rows)
    
    for i in 1:n_rows
        row = df[i, :]
        
        # entani論文の重み
        entani_wᴸ[i] = parse_weight_vector(row.entani論文のPerfectIncorporation_wᴸ)
        entani_wᵁ[i] = parse_weight_vector(row.entani論文のPerfectIncorporation_wᵁ)
        
        # 解の非唯一性考慮の重み
        nonunique_wᴸ[i] = parse_weight_vector(row.解の非唯一性考慮のPerfectIncorporation_wᴸ)
        nonunique_wᵁ[i] = parse_weight_vector(row.解の非唯一性考慮のPerfectIncorporation_wᵁ)
        
        # 各DMの重みを取得
        entani_wᴸ_dm1 = parse_weight_vector(row.entani論文のPerfectIncorporation_wᴸ_1)
        entani_wᵁ_dm1 = parse_weight_vector(row.entani論文のPerfectIncorporation_wᵁ_1)
        entani_wᴸ_dm2 = parse_weight_vector(row.entani論文のPerfectIncorporation_wᴸ_2)
        entani_wᵁ_dm2 = parse_weight_vector(row.entani論文のPerfectIncorporation_wᵁ_2)
        entani_wᴸ_dm3 = parse_weight_vector(row.entani論文のPerfectIncorporation_wᴸ_3)
        entani_wᵁ_dm3 = parse_weight_vector(row.entani論文のPerfectIncorporation_wᵁ_3)
        
        nonunique_wᴸ_dm1 = parse_weight_vector(row.解の非唯一性考慮のPerfectIncorporation_wᴸ_1)
        nonunique_wᵁ_dm1 = parse_weight_vector(row.解の非唯一性考慮のPerfectIncorporation_wᵁ_1)
        nonunique_wᴸ_dm2 = parse_weight_vector(row.解の非唯一性考慮のPerfectIncorporation_wᴸ_2)
        nonunique_wᵁ_dm2 = parse_weight_vector(row.解の非唯一性考慮のPerfectIncorporation_wᵁ_2)
        nonunique_wᴸ_dm3 = parse_weight_vector(row.解の非唯一性考慮のPerfectIncorporation_wᴸ_3)
        nonunique_wᵁ_dm3 = parse_weight_vector(row.解の非唯一性考慮のPerfectIncorporation_wᵁ_3)
        
        # 行列に変換して保存（有効なデータがある場合のみ）
        if !isempty(entani_wᴸ_dm1) && !isempty(entani_wᴸ_dm2) && !isempty(entani_wᴸ_dm3)
            entani_wᴸ_dms[i] = hcat(entani_wᴸ_dm1, entani_wᴸ_dm2, entani_wᴸ_dm3)
            entani_wᵁ_dms[i] = hcat(entani_wᵁ_dm1, entani_wᵁ_dm2, entani_wᵁ_dm3)
        end
        
        if !isempty(nonunique_wᴸ_dm1) && !isempty(nonunique_wᴸ_dm2) && !isempty(nonunique_wᴸ_dm3)
            nonunique_wᴸ_dms[i] = hcat(nonunique_wᴸ_dm1, nonunique_wᴸ_dm2, nonunique_wᴸ_dm3)
            nonunique_wᵁ_dms[i] = hcat(nonunique_wᵁ_dm1, nonunique_wᵁ_dm2, nonunique_wᵁ_dm3)
        end
    end
    
    return entani_wᴸ, entani_wᵁ, nonunique_wᴸ, nonunique_wᵁ, 
           entani_wᴸ_dms, entani_wᵁ_dms, nonunique_wᴸ_dms, nonunique_wᵁ_dms
end

# 区間重要度から区間PCMを計算する関数
function calculate_interval_pcm(wᴸ::Vector{Float64}, wᵁ::Vector{Float64})
    n = length(wᴸ)
    pcm_L = zeros(Float64, n, n)
    pcm_U = zeros(Float64, n, n)
    
    for i in 1:n
        for j in 1:n
            if i == j
                # 対角成分は[1,1]に設定
                pcm_L[i, j] = 1.0
                pcm_U[i, j] = 1.0
            else
                # 区間PCMの下限 = wiᴸ / wjᵁ
                pcm_L[i, j] = wᴸ[i] / wᵁ[j]
                # 区間PCMの上限 = wiᵁ / wjᴸ
                pcm_U[i, j] = wᵁ[i] / wᴸ[j]
            end
        end
    end
    
    return pcm_L, pcm_U
end

# PCM行列を文字列表現に変換する関数
function pcm_to_string(pcm::Matrix{Float64})
    rows, cols = size(pcm)
    result = "["
    
    for i in 1:rows
        result *= "["
        for j in 1:cols
            result *= string(round(pcm[i, j], digits=4))
            if j < cols
                result *= ", "
            end
        end
        result *= "]"
        if i < rows
            result *= ", "
        end
    end
    
    result *= "]"
    return result
end

function process_and_save_interval_pcms_perfect(input_file::String, output_file::String)
    # CSVファイルを読み込む
    df = CSV.read(input_file, DataFrame)
    
    # "entani論文のPerfectIncorporation_wᴸ" 列に数値が存在する行の "Trial" 列を選択
    valid_rows = .!ismissing.(df[:, :"entani論文のPerfectIncorporation_wᴸ"]) .&
                .!(df[:, :"entani論文のPerfectIncorporation_wᴸ"] .== "E")
                
    # Trial番号を抽出
    unique_trial_numbers = unique(df[valid_rows, :"Trial"])
    
    # 有効なデータを抽出
    relevant_data = extract_interval_weights(input_file, unique_trial_numbers)
    
    # 区間重要度を抽出
    entani_wᴸ, entani_wᵁ, nonunique_wᴸ, nonunique_wᵁ, 
    entani_wᴸ_dms, entani_wᵁ_dms, nonunique_wᴸ_dms, nonunique_wᵁ_dms = 
        convert_weights_to_matrices(relevant_data)
    
    # 結果を格納するDataFrame
    results_df = DataFrame(
        Trial = Int[],
        Entani統合PCM_L = String[],
        Entani統合PCM_U = String[],
        非唯一性統合PCM_L = String[],
        非唯一性統合PCM_U = String[],
        Entani_DM1_PCM_L = String[],
        Entani_DM1_PCM_U = String[],
        Entani_DM2_PCM_L = String[],
        Entani_DM2_PCM_U = String[],
        Entani_DM3_PCM_L = String[],
        Entani_DM3_PCM_U = String[],
        非唯一性_DM1_PCM_L = String[],
        非唯一性_DM1_PCM_U = String[],
        非唯一性_DM2_PCM_L = String[],
        非唯一性_DM2_PCM_U = String[],
        非唯一性_DM3_PCM_L = String[],
        非唯一性_DM3_PCM_U = String[]
    )
    
    # 各トライアルごとに区間PCMを計算
    for i in 1:length(unique_trial_numbers)
        trial = unique_trial_numbers[i]
        
        # 区間ウェイトが有効であることを確認
        if isassigned(entani_wᴸ, i) && !isempty(entani_wᴸ[i]) && isassigned(entani_wᵁ, i) && !isempty(entani_wᵁ[i])
            # Entani統合PCM
            entani_pcm_L, entani_pcm_U = calculate_interval_pcm(entani_wᴸ[i], entani_wᵁ[i])
            entani_pcm_L_str = pcm_to_string(entani_pcm_L)
            entani_pcm_U_str = pcm_to_string(entani_pcm_U)
            
            # 各DMのPCM (Entani)
            entani_dm1_pcm_L, entani_dm1_pcm_U = calculate_interval_pcm(
                entani_wᴸ_dms[i][:, 1], entani_wᵁ_dms[i][:, 1])
            entani_dm2_pcm_L, entani_dm2_pcm_U = calculate_interval_pcm(
                entani_wᴸ_dms[i][:, 2], entani_wᵁ_dms[i][:, 2])
            entani_dm3_pcm_L, entani_dm3_pcm_U = calculate_interval_pcm(
                entani_wᴸ_dms[i][:, 3], entani_wᵁ_dms[i][:, 3])
            
            entani_dm1_pcm_L_str = pcm_to_string(entani_dm1_pcm_L)
            entani_dm1_pcm_U_str = pcm_to_string(entani_dm1_pcm_U)
            entani_dm2_pcm_L_str = pcm_to_string(entani_dm2_pcm_L)
            entani_dm2_pcm_U_str = pcm_to_string(entani_dm2_pcm_U)
            entani_dm3_pcm_L_str = pcm_to_string(entani_dm3_pcm_L)
            entani_dm3_pcm_U_str = pcm_to_string(entani_dm3_pcm_U)
        else
            entani_pcm_L_str = ""
            entani_pcm_U_str = ""
            entani_dm1_pcm_L_str = ""
            entani_dm1_pcm_U_str = ""
            entani_dm2_pcm_L_str = ""
            entani_dm2_pcm_U_str = ""
            entani_dm3_pcm_L_str = ""
            entani_dm3_pcm_U_str = ""
        end
        
        # 非唯一性区間PCM
        if isassigned(nonunique_wᴸ, i) && !isempty(nonunique_wᴸ[i]) && isassigned(nonunique_wᵁ, i) && !isempty(nonunique_wᵁ[i])
            nonunique_pcm_L, nonunique_pcm_U = calculate_interval_pcm(nonunique_wᴸ[i], nonunique_wᵁ[i])
            nonunique_pcm_L_str = pcm_to_string(nonunique_pcm_L)
            nonunique_pcm_U_str = pcm_to_string(nonunique_pcm_U)
            
            # 各DMのPCM (非唯一性)
            nonunique_dm1_pcm_L, nonunique_dm1_pcm_U = calculate_interval_pcm(
                nonunique_wᴸ_dms[i][:, 1], nonunique_wᵁ_dms[i][:, 1])
            nonunique_dm2_pcm_L, nonunique_dm2_pcm_U = calculate_interval_pcm(
                nonunique_wᴸ_dms[i][:, 2], nonunique_wᵁ_dms[i][:, 2])
            nonunique_dm3_pcm_L, nonunique_dm3_pcm_U = calculate_interval_pcm(
                nonunique_wᴸ_dms[i][:, 3], nonunique_wᵁ_dms[i][:, 3])
            
            nonunique_dm1_pcm_L_str = pcm_to_string(nonunique_dm1_pcm_L)
            nonunique_dm1_pcm_U_str = pcm_to_string(nonunique_dm1_pcm_U)
            nonunique_dm2_pcm_L_str = pcm_to_string(nonunique_dm2_pcm_L)
            nonunique_dm2_pcm_U_str = pcm_to_string(nonunique_dm2_pcm_U)
            nonunique_dm3_pcm_L_str = pcm_to_string(nonunique_dm3_pcm_L)
            nonunique_dm3_pcm_U_str = pcm_to_string(nonunique_dm3_pcm_U)
        else
            nonunique_pcm_L_str = ""
            nonunique_pcm_U_str = ""
            nonunique_dm1_pcm_L_str = ""
            nonunique_dm1_pcm_U_str = ""
            nonunique_dm2_pcm_L_str = ""
            nonunique_dm2_pcm_U_str = ""
            nonunique_dm3_pcm_L_str = ""
            nonunique_dm3_pcm_U_str = ""
        end
        
        # 結果をDataFrameに追加
        push!(results_df, (
            Trial = trial,
            Entani統合PCM_L = entani_pcm_L_str,
            Entani統合PCM_U = entani_pcm_U_str,
            非唯一性統合PCM_L = nonunique_pcm_L_str,
            非唯一性統合PCM_U = nonunique_pcm_U_str,
            Entani_DM1_PCM_L = entani_dm1_pcm_L_str,
            Entani_DM1_PCM_U = entani_dm1_pcm_U_str,
            Entani_DM2_PCM_L = entani_dm2_pcm_L_str,
            Entani_DM2_PCM_U = entani_dm2_pcm_U_str,
            Entani_DM3_PCM_L = entani_dm3_pcm_L_str,
            Entani_DM3_PCM_U = entani_dm3_pcm_U_str,
            非唯一性_DM1_PCM_L = nonunique_dm1_pcm_L_str,
            非唯一性_DM1_PCM_U = nonunique_dm1_pcm_U_str,
            非唯一性_DM2_PCM_L = nonunique_dm2_pcm_L_str,
            非唯一性_DM2_PCM_U = nonunique_dm2_pcm_U_str,
            非唯一性_DM3_PCM_L = nonunique_dm3_pcm_L_str,
            非唯一性_DM3_PCM_U = nonunique_dm3_pcm_U_str
        ))
    end
    
    # 結果をCSVファイルに書き出し
    CSV.write(output_file, results_df)
    println("区間PCMの計算が完了し、$(output_file)に保存されました。")
end

"""
CSVファイルから区間PCMを読み込み、指定された条件に基づいて表示する関数
引数:
- csv_file: CSVファイルパス
- trial_number: 表示するトライアル番号
- method: 表示する手法名（"Entani", "非唯一性", "All"）
- dm_number: 表示する意思決定者番号（1, 2, 3, 0(統合PCM), -1(すべて)）
"""
function display_filtered_interval_pcm(csv_file::String, trial_number::Int; method::String="All", dm_number::Int=-1)
    # CSVファイルを読み込む
    df = CSV.read(csv_file, DataFrame)
    
    # 指定されたtrial番号の行を取得
    row = filter(row -> row.Trial == trial_number, df)
    
    if nrow(row) == 0
        println("Trial $(trial_number)のデータが見つかりませんでした。")
        return
    end
    
    # 対象のtrial行を取得
    trial_row = first(row)
    
    # フィルタリング条件に基づいてPCM行列のリストを作成
    pcm_pairs = []
    
    # 手法名に基づくフィルタリング
    methods_to_display = []
    if method == "All"
        methods_to_display = ["Entani", "非唯一性"]
    else
        methods_to_display = [method]
    end
    
    # 意思決定者番号に基づくフィルタリング
    dm_numbers_to_display = []
    if dm_number == -1  # すべての意思決定者
        dm_numbers_to_display = [0, 1, 2, 3]  # 0は統合PCM
    else
        dm_numbers_to_display = [dm_number]
    end
    
    # フィルタリング条件に合致するPCMペアを選択
    for m in methods_to_display
        for dm in dm_numbers_to_display
            if dm == 0  # 統合PCM
                if m == "Entani"
                    push!(pcm_pairs, ("Entani統合PCM", trial_row.Entani統合PCM_L, trial_row.Entani統合PCM_U))
                elseif m == "非唯一性"
                    push!(pcm_pairs, ("非唯一性統合PCM", trial_row.非唯一性統合PCM_L, trial_row.非唯一性統合PCM_U))
                end
            else  # 個別DM
                if m == "Entani"
                    col_L = Symbol("Entani_DM$(dm)_PCM_L")
                    col_U = Symbol("Entani_DM$(dm)_PCM_U")
                    if hasproperty(trial_row, col_L) && hasproperty(trial_row, col_U)
                        push!(pcm_pairs, ("Entani_DM$(dm)_PCM", trial_row[col_L], trial_row[col_U]))
                    end
                elseif m == "非唯一性"
                    col_L = Symbol("非唯一性_DM$(dm)_PCM_L")
                    col_U = Symbol("非唯一性_DM$(dm)_PCM_U")
                    if hasproperty(trial_row, col_L) && hasproperty(trial_row, col_U)
                        push!(pcm_pairs, ("非唯一性_DM$(dm)_PCM", trial_row[col_L], trial_row[col_U]))
                    end
                end
            end
        end
    end
    
    # 選択されたPCMを表示
    if isempty(pcm_pairs)
        println("指定された条件に合致するPCMデータがありません。")
        return
    end
    
    for (name, pcm_L_str, pcm_U_str) in pcm_pairs
        if !isempty(pcm_L_str) && !isempty(pcm_U_str)
            println("\n### $(name) (Trial $(trial_number))")
            
            # 文字列からPCM行列を解析
            pcm_L = parse_matrix_string_improved(pcm_L_str)
            pcm_U = parse_matrix_string_improved(pcm_U_str)
            
            # 解析に成功した場合のみ表示
            if !isempty(pcm_L) && !isempty(pcm_U)
                # 区間PCM行列をLaTeX形式で表示
                display_interval_pcm_latex(pcm_L, pcm_U)
                
                # 数値表形式で表示
                display_interval_pcm_table(pcm_L, pcm_U)
            end
        end
    end
end

"""
文字列からPCM行列を解析する関数（改良版）
入れ子配列を適切に処理
"""
function parse_matrix_string_improved(matrix_str::String)
    if isempty(matrix_str)
        return Array{Float64}(undef, 0, 0)
    end
    
    try
        # 配列構造を評価
        raw_data = eval(Meta.parse(matrix_str))
        
        # 行数と列数を決定
        num_rows = length(raw_data)
        num_cols = length(raw_data[1])
        
        # 配列を初期化
        matrix = Array{Float64}(undef, num_rows, num_cols)
        
        # データを転送
        for i in 1:num_rows
            for j in 1:num_cols
                matrix[i, j] = raw_data[i][j]
            end
        end
        
        return matrix
    catch e
        println("行列の解析中にエラーが発生しました: $e")
        println("問題の文字列: $matrix_str")
        return Array{Float64}(undef, 0, 0)
    end
end

"""
区間PCM行列をLaTeX形式で表示する関数
"""
function display_interval_pcm_latex(pcm_L::Array{Float64,2}, pcm_U::Array{Float64,2})
    n = size(pcm_L, 1)
    
    # 行列の次元を確認
    if n == 0 || size(pcm_U, 1) == 0
        println("有効な行列データがありません。")
        return
    end
    
    # LaTeX形式で行列を表示
    println("\\begin{bmatrix}")
    
    for i in 1:n
        for j in 1:n
            print("[$(round(pcm_L[i,j], digits=4)), $(round(pcm_U[i,j], digits=4))]")
            if j < n
                print(" & ")
            end
        end
        
        if i < n
            println(" \\\\")
        else
            println("")
        end
    end
    
    println("\\end{bmatrix}")
    println()
end

"""
区間PCM行列を数値表形式で表示する関数
"""
function display_interval_pcm_table(pcm_L::Array{Float64,2}, pcm_U::Array{Float64,2})
    n = size(pcm_L, 1)
    
    # 行列の次元を確認
    if n == 0 || size(pcm_U, 1) == 0
        println("有効な行列データがありません。")
        return
    end
    
    # 表のヘッダー
    print("           ")
    for j in 1:n
        print("項目$(j)                ")
    end
    println()
    
    # 表の内容
    for i in 1:n
        print("項目$(i)    ")
        for j in 1:n
            print("[$(round(pcm_L[i,j], digits=4)), $(round(pcm_U[i,j], digits=4))]    ")
        end
        println()
    end
    println()
end

"""
特定の区間PCM行列を直接表示する関数
"""
function display_specific_pcm(L_str::String, U_str::String, name::String = "区間PCM行列")
    pcm_L = parse_matrix_string_improved(L_str)
    pcm_U = parse_matrix_string_improved(U_str)
    
    if !isempty(pcm_L) && !isempty(pcm_U)
        println("\n### $(name)")
        display_interval_pcm_latex(pcm_L, pcm_U)
        display_interval_pcm_table(pcm_L, pcm_U)
    else
        println("行列の解析に失敗しました。")
    end
end

# 以下に使用例を示します

# 例1: Trial 1のすべての区間PCMを表示
function example1()
    println("===== 例1: Trial 1のすべての区間PCM =====")
    display_filtered_interval_pcm(csv_file, 1)
end

# 例2: Trial 5のEntani手法のみ表示
function example2()
    println("===== 例2: Trial 5のEntani手法のみ =====")
    display_filtered_interval_pcm(csv_file, 5, method="Entani")
end

# 例3: Trial 10の意思決定者1のPCMのみ表示
function example3()
    println("===== 例3: Trial 10の意思決定者1のPCMのみ =====")
    display_filtered_interval_pcm(csv_file, 10, dm_number=1)
end

# 例4: Trial 3のEntani手法の意思決定者2のPCMのみ表示
function example4()
    println("===== 例4: Trial 3のEntani手法の意思決定者2のPCM =====")
    display_filtered_interval_pcm(csv_file, 3, method="Entani", dm_number=2)
end

# 例5: Trial 8の統合PCMのみ表示
function example5()
    println("===== 例5: Trial 8の統合PCMのみ =====")
    display_filtered_interval_pcm(csv_file, 8, dm_number=0)
end

# 例6: 与えられた区間PCMを直接表示
function example6()
    println("===== 例6: 指定された区間PCMを直接表示 =====")
    L_str = "[[0.6328, 1.0323, 1.0, 1.4915, 1.4915], [0.25, 0.4078, 0.3951, 0.5893, 0.5893], [0.4242, 0.6921, 0.6705, 1.0, 1.0], [0.1582, 0.2581, 0.25, 0.3729, 0.3729], [0.0833, 0.1359, 0.1317, 0.1964, 0.1964]]"
    U_str = "[[1.5804, 4.0, 2.3571, 6.3214, 12.0], [0.9687, 2.452, 1.4449, 3.875, 7.3559], [1.0, 2.5311, 1.4915, 4.0, 7.5932], [0.6705, 1.697, 1.0, 2.6818, 5.0909], [0.6705, 1.697, 1.0, 2.6818, 5.0909]]"
    display_specific_pcm(L_str, U_str, "指定された区間PCM行列")
end

# メイン関数: 使用例を実行
function main()
    # 使用例の実行（必要な例のみコメントを解除）
    # example1()  # すべての区間PCM
    # example2()  # Entani手法のみ
    # example3()  # 意思決定者1のみ
    # example4()  # Entani手法の意思決定者2のみ
    # example5()  # 統合PCMのみ
    example6()  # 指定された区間PCMを直接表示
end
