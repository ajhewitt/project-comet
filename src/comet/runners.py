def run(args) -> int:
    # pretend to run a pipeline; honor --ordering if present
    ordering = getattr(args, "ordering", "both")
    if ordering not in {"AtoB", "BtoA", "both"}:
        raise ValueError("invalid ordering")
    return 0
