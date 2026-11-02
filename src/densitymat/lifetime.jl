# Deferred, ordered destruction of library objects reclaimed by the GC.
#
# cuDensityMat objects reference each other (an operator uses its terms, a term its
# elementary operators, everything the library handle). Julia does not order finalizers:
# objects that become unreachable together are finalized in arbitrary order, and an
# explicit `close(ws)` used to destroy the library handle while states/operators created
# from it were still alive. Destroying a parent after its children (or anything after the
# handle) is a use-after-free inside the library — on the A10 it corrupted the host heap
# (`malloc_consolidate(): unaligned fastbin chunk detected`) once a GC-heavy test ran
# finalizers of earlier testsets' objects.
#
# Finalizers therefore never call the library: they queue a destroy thunk with a rank,
# and `drain_pending_destroys!()` (run at safe points — every `_check_valid(ws)` and
# `close(ws)`) destroys the queued objects users-first: actions / expectations / spectra
# (1) → operators (2) → terms (3) → elementary / matrix operators (4) → states (5) →
# workspace descriptor + handle (6). A queued object is unreachable, so every object that
# uses it is unreachable too and was queued no later than it; ranking makes the order safe.
# Explicit `destroy_*` calls still destroy immediately (caller's responsibility).

const _PENDING_LOCK = ReentrantLock()
const _PENDING = Tuple{Int,Any,UInt}[]          # (rank, destroy thunk, owning library handle or 0)
const _GRAVEYARD = Dict{UInt,Vector{Tuple{Int,Any}}}()

"""
When `true` (default), GC-reclaimed objects are destroyed only when their `WorkStream`
is closed or collected, never while it is still in use. On the A10 (libcudensitymat
0.7.0) destroying a prepared operator and then building/preparing another one on the
same library handle aborted inside the library ("TensorNetwork::placeTensor:
Connections are invalid") or hung; deferring the destroys to the end of the
WorkStream's life avoids it at the cost of keeping the dead objects' (host-side)
library structures until then. Set `CuDensityMat.DEFER_DESTROY_UNTIL_CLOSE[] = false`
to destroy as soon as possible (in dependency order).
"""
const DEFER_DESTROY_UNTIL_CLOSE = Ref(true)

_owner_key(obj) =
    hasproperty(obj, :ws) && obj.ws isa WorkStream ? UInt(obj.ws.handle) : UInt(0)

# Called from finalizers only: no blocking lock (the documented trylock / re-register pattern).
function _defer_destroy(obj, rank::Int, thunk, key::UInt = _owner_key(obj))
    if trylock(_PENDING_LOCK)
        try
            push!(_PENDING, (rank, thunk, key))
        finally
            unlock(_PENDING_LOCK)
        end
    else
        finalizer(o -> _defer_destroy(o, rank, thunk, key), obj)
    end
    return nothing
end

_run_sorted!(items) = (sort!(items; by = first); foreach(x -> x[2](), items); length(items))

"""
    drain_pending_destroys!() -> Int

Destroy the library objects whose Julia wrappers were garbage collected, in dependency
order (see the comment at the top of `lifetime.jl`); with
[`DEFER_DESTROY_UNTIL_CLOSE`](@ref) objects of a live WorkStream are parked until that
WorkStream is closed or collected. Returns the number destroyed. Runs automatically at
every API entry that validates a `WorkStream`.
"""
function drain_pending_destroys!()
    isempty(_PENDING) && return 0
    items = lock(_PENDING_LOCK) do
        x = copy(_PENDING)
        empty!(_PENDING)
        x
    end
    now = Tuple{Int,Any}[]
    for (rank, f, key) in items
        if rank == 6
            # the WorkStream itself: its parked objects first, then the handle
            append!(now, get(_GRAVEYARD, key, Tuple{Int,Any}[]))
            delete!(_GRAVEYARD, key)
            delete!(_LIVE_HANDLES, key)
            push!(now, (rank, f))
        elseif DEFER_DESTROY_UNTIL_CLOSE[] && key != 0 && _live_handle(key)
            push!(get!(Vector{Tuple{Int,Any}}, _GRAVEYARD, key), (rank, f))
        else
            push!(now, (rank, f))
        end
    end
    return _run_sorted!(now)
end

# Handles of WorkStreams that are neither closed nor collected.
const _LIVE_HANDLES = Set{UInt}()
_live_handle(key) = key in _LIVE_HANDLES

"""Destroy the objects parked for the WorkStream with library handle `key` (it is closing)."""
function _bury!(key::UInt)
    delete!(_LIVE_HANDLES, key)
    drain_pending_destroys!()
    items = pop!(_GRAVEYARD, key, Tuple{Int,Any}[])
    return _run_sorted!(items)
end
