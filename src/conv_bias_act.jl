export conv_bias_act, conv_bias_act!

function conv_bias_act(x::AbstractArray{xT,N}, w::AbstractArray{wT,N},
                cdims::ConvDims, b=false, σ=identity; kwargs...) where {xT, wT, N}
    y = similar(x, promote_type(xT, wT, bT), output_size(cdims)..., channels_out(cdims), size(x,N))
    conv_bias_act!(y, x, w, cdims, b, σ; kwargs...)
    return y
end

function conv_bias_act!(y::AbstractArray{yT,5}, x::AbstractArray{xT,5}, w::AbstractArray{wT,5},
                cdims::ConvDims, b=false, σ=identity; kwargs...) where {yT, xT, wT}
    conv!(y, x, w, cdims)
    y .= σ.(y .+ b)
    return y
end

for N in (3, 4)
    @eval begin
        function conv_bias_act(
                        y::AbstractArray{yT,$N}, x::AbstractArray{xT,$N},
                        w::AbstractArray{wT,$N}, cdims::ConvDims,
                        b=false, σ=identity;
                        kwargs...) where {yT, xT, wT}
            conv_bias_act!(
                insert_singleton_spatial_dimension(y, $(5 - N)),
                insert_singleton_spatial_dimension(x, $(5 - N)),
                insert_singleton_spatial_dimension(w, $(5 - N)),
                insert_singleton_spatial_dimension(cdims, $(5 - N)),
                insert_singleton_spatial_dimension(b, $(5 - N)),
                σ;
                kwargs...
            )

            # We explicitly return `y` here, because the backend call
            # itself may return a reshaped view, which we don't want.
            return y
        end
    end
end

for sigma in [:identity, :relu, :tanh]
    # For these activation functions, we can calculate the gradient using only the final y

    @eval function ChainRulesCore.rrule(::typeof(conv_bias_act), x, w, cdims, b, σ::typeof($sigma); kw...)
        y = conv_bias_act(x, w, cdims, b, σ; kw...)
        function conv_bias_act_pullback(Δ)
            # First you need to un-broadcast the activation??
            Δσ = if σ === identity
                colmajor(Δ) # copy it
            elseif σ === relu
                similar(Δ) .= y .> 0
            elseif σ === tanh
                # ??
                error()
            end

            # Second, you need to project onto the bias vector
            db = if b===false
                DoesNotExist()
            else
                sum!(similar(b), Δσ) # probably needs some reshaping first
            end

            # Third, you work out the nontrivial bits
            return (
                NO_FIELDS,
                @thunk(∇conv_data(Δσ, w, cdims, kw...)),
                @thunk(∇conv_filter(x, Δσ, cdims, kw...)),
                DoesNotExist(),
                db,
                DoesNotExist(),
            )
        end
        return y, conv_bias_act_pullback
    end

end

#=
for conv in [:conv, :depthwiseconv]
    local ∇conv_data, ∇conv_filter = Symbol.(:∇, conv, [:_data, :_filter])
    conv_pullback, ∇conv_data_pullback = Symbol.([conv, ∇conv_data], :_pullback)

    @eval function ChainRulesCore.rrule(::typeof($conv), x, w, cdims; kw...)
        function $conv_pullback(Δ)
            Δ = colmajor(Δ)
            return (
                NO_FIELDS,
                @thunk($∇conv_data(Δ, w, cdims, kw...)),
                @thunk($∇conv_filter(x, Δ, cdims, kw...)),
                DoesNotExist(),
            )
        end
        return $conv(x, w, cdims; kw...), $conv_pullback
    end

    @eval function ChainRulesCore.rrule(::typeof($∇conv_data), x, w, cdims; kw...)
        function $∇conv_data_pullback(Δ)
            Δ = colmajor(Δ)
            return (
                NO_FIELDS,
                @thunk($conv(Δ, w, cdims, kw...)),
                @thunk($∇conv_filter(Δ, x, cdims, kw...)),
                DoesNotExist(),
            )
        end
        return $∇conv_data(x, w, cdims; kw...), $∇conv_data_pullback
    end
end
=#
