# Evaluate Cal Miller's tracer/ParticleTracing.jl WITHOUT running it.
#
# The vendored file (upstream 543b42e, 16 approved diffs) is not
# copied or forked by this loader. It is read from disk and evaluated here with
# exactly one line removed -- its own trailing `allstats = main(args)` -- so
# the includer, not the vendored file, decides when a run starts.  Everything
# else executes verbatim: the ArgParse table (identical flags, defaults and
# footguns), every constant, every physics function.  A byte-identity
# harness in the private development repository compares historical references; controls now follow diff 16.
#
# The one-line removal is asserted, not assumed: if the entry point is not
# found exactly once this errors out rather than silently loading a file whose
# shape has changed.

const _VENDORED = joinpath(@__DIR__, "ParticleTracing.jl")
const _ENTRY = "allstats = main(args)"

let src = read(_VENDORED, String), hits = findall(_ENTRY, src)
    length(hits) == 1 || error("vendored.jl: expected exactly one `$_ENTRY` " *
        "in $_VENDORED, found $(length(hits)); the vendored file changed " *
        "shape, re-read it before trusting this loader")
    include_string(@__MODULE__,
        replace(src, _ENTRY => "# entry point removed by tracer/vendored.jl"),
        _VENDORED)
end
