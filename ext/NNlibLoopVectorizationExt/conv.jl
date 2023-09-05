#=

Implementation copied from here (Jonas Steinebach, MIT):
https://github.com/jonas208/GradValley.jl/blob/main/src/functional/gv_convolution.jl

Almost certainly wrong due to Conv / CrossCorr conventions.
Could include bias & activation too, hence overload `conv_bias_act`,
at the cost of needing gradient rules for that.

=#

# println("loaded extension")

function NNlib.conv!(output::Array{T,4}, input::Array{T,4}, weight::Array{T,4}, cdims::ConvDims; kw...) where {T<:Float32}

    if cdims.padding != (0, 0, 0, 0)
        invoke(NNlib.conv!, 
            Tuple{AbstractArray{T,4},AbstractArray{T,4},AbstractArray{T,4},ConvDims}, 
            output, input, weight, cdims; kw...)
    end

    output_width, output_height, _ = size(output)
    input_width, input_height, in_channels, batches = size(input)
    weight_width, weight_height, in_channels_weight, out_channels = size(weight)

    groups = cdims.groupcount
    y_stride, x_stride = cdims.stride
    y_dilation, x_dilation = cdims.dilation
    out_channels_per_group = out_channels ÷ groups

    if cdims.groupcount == 1 && cdims.stride == (1, 1) && cdims.dilation == (1, 1)
        # println("very specialized case for maximum performance")

        @tturbo for index_batch in 1:batches
            for out_channel in 1:out_channels, y_out in 1:output_height, x_out in 1:output_width
                value = zero(T)
                for in_channel in 1:in_channels, y_w in 1:weight_height, x_w in 1:weight_width
                    value += input[x_out + x_w - 1, y_out + y_w - 1, in_channel, index_batch] * weight[x_w, y_w, in_channel, out_channel]
                end
                output[x_out, y_out, out_channel, index_batch] = value #+ bias[out_channel]
            end
        end

    elseif groups == 1
        # println("second specialized case for better performance")

        @tturbo for index_batch in 1:batches
            for out_channel in 1:out_channels, y_out in 1:output_height, x_out in 1:output_width
                m = y_out + (y_stride - 1) * (y_out - 1)
                n = x_out + (x_stride - 1) * (x_out - 1)
                value = zero(T)
                for in_channel in 1:in_channels, y_w in 1:weight_height, x_w in 1:weight_width
                    y_in = m + (y_w - 1) * y_dilation
                    x_in = n + (x_w - 1) * x_dilation
                    value += input[x_in, y_in, in_channel, index_batch] * weight[x_w, y_w, in_channel, out_channel]
                end
                output[x_out, y_out, out_channel, index_batch] = value #+ bias[out_channel]
            end
        end

    else 
        # println("general case for any convolution")

        @tturbo for index_batch in 1:batches
            for group in 1:groups, out_channel_per_group in 1:out_channels_per_group, y_out in 1:output_height, x_out in 1:output_width
                m = y_out + (y_stride - 1) * (y_out - 1)
                n = x_out + (x_stride - 1) * (x_out - 1)
                out_channel = (group * out_channels_per_group + 1) - out_channel_per_group
                value = zero(T)
                for in_channel_weight in 1:in_channels_weight, y_w in 1:weight_height, x_w in 1:weight_width
                    y_in = m + (y_w - 1) * y_dilation
                    x_in = n + (x_w - 1) * x_dilation
                    in_channel_input = in_channel_weight + (group - 1) * in_channels_weight
                    value += input[x_in, y_in, in_channel_input, index_batch] * weight[x_w, y_w, in_channel_weight, out_channel]
                end
                output[x_out, y_out, out_channel, index_batch] = value #+ bias[out_channel]
            end
        end
    end

    return output
end

#=

julia> lenet = Chain(
           Conv((5, 5), 1=>6, relu),
           MaxPool((2, 2)),
           Conv((5, 5), 6=>16, relu),
           MaxPool((2, 2)),
           Flux.flatten,
           Dense(256 => 120, relu),
           Dense(120 => 84, relu), 
           Dense(84 => 10),
       );

julia> img = rand32(28, 28, 1, 128);

julia> @btime $lenet($img);
  min 930.417 μs, mean 1.370 ms (160 allocations, 5.60 MiB)

julia> using LoopVectorization

julia> @btime $lenet($img);
  min 555.417 μs, mean 1.276 ms (56 allocations, 5.13 MiB)

=#

