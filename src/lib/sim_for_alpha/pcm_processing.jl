
using CSV
using DataFrames
using IntervalArithmetic
using JuMP
import HiGHS

# Include the provided functions
include("../crisp-pcm.jl")
include("../nearly-equal.jl")
include("../ttimes/optimal-value.jl")

# Function to calculate t_L for each PCM
function calculate_t_L(pcm::Matrix{Float64})
    n = size(pcm, 1)
    w_center1 = solveCrispAHPLP(pcm)
    candidate = zeros(Float64, n)

    for l = 1:n
        wᵢᴸ_check = w_center1.wᴸ_center_1[l]
        ∑wⱼᵁ = sum(map(j -> w_center1.wᵁ_center_1[j], filter(j -> l != j, 1:n)))
        candidate[l] = ∑wⱼᵁ + wᵢᴸ_check
    end

    t_L = 1 / minimum(candidate)
    return t_L
end

# Function to calculate t_L for each PCM
function calculate_t_U(pcm::Matrix{Float64})
    n = size(pcm, 1)
    w_center1 = solveCrispAHPLP(pcm)
    candidate2 = zeros(Float64, n)
    
    for l = 1:n
        wᵢᵁ_check = w_center1.wᵁ_center_1[l]
        ∑wⱼᴸ = sum(map(j -> w_center1.wᴸ_center_1[j], filter(j -> l != j, 1:n)))
        candidate2[l] = ∑wⱼᴸ + wᵢᵁ_check
    end

    t_U = 1 / maximum(candidate2)
    return t_U
end

# Main process
function process_csv(input_file::String, output_file::String)
    df = CSV.read(input_file, DataFrame)
    results = DataFrame(Trial = Int[], PCM = Int[], t_L = Float64[], t_U = Float64[])

    # Assuming each PCM is recorded in 5 consecutive rows
    for i in 1:5:size(df, 1)
        pcm_data = df[i:i+4, :]
        # Assuming that pcm data is in columns 3 to 7
        pcm = Matrix(pcm_data[:, 3:7])
        t_L = calculate_t_L(pcm)
        t_U = calculate_t_U(pcm)
        # Assuming that Trial and PCM numbers are in the first and second columns
        trial = pcm_data[1, 1]
        pcm_number = pcm_data[1, 2]
        push!(results, (trial, pcm_number, t_L, t_U))
    end

    CSV.write(output_file, results)
end
