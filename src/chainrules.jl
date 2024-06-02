function ChainRulesCore.rrule(::typeof(maximin), input, minmax)
    Y = maximin(input, minmax)
    function maximin_pullback(Ȳ)
        ∂input = @thunk(@views @.  Ȳ / (minmax[:,2] - minmax[:,1]))
        return NoTangent(), ∂input, NoTangent()
    end
    return Y, maximin_pullback
end

function ChainRulesCore.rrule(::typeof(inv_maximin), input, minmax)
    Y = inv_maximin(input, minmax)
    function inv_maximin_pullback(Ȳ)
        ∂input = @thunk(@views @.  Ȳ * (minmax[:,2] - minmax[:,1]))
        return NoTangent(), ∂input, NoTangent()
    end
    return Y, inv_maximin_pullback
end
