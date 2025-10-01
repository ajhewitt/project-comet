def fetch(args) -> int:
    # placeholder data fetch; respects args.action for future expansion
    action = getattr(args, "action", "fetch")
    if action != "fetch":
        raise ValueError("unsupported action")
    return 0
