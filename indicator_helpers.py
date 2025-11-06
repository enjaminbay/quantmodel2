def split_indicator_name(name: str) -> tuple:
    """Split indicator name into base indicator and type"""
    parts = name.split('_')
    if len(parts) == 1:
        return parts[0], 'base'
    elif parts[-1] in ['derivative', 'acceleration']:
        return '_'.join(parts[:-1]), parts[-1]
    return name, 'base'