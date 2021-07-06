from pathlib import Path


def resolve_path(path_str: str) -> Path:
    """Creates an absolute Path object from a path string

    Args:
        path_str (str): The path string

    Returns:
        path (Path): The created Path object
    """

    _dirs = path_str.split("/")
    path = None

    if path_str[0] == "/":
        path = Path(path_str)
    elif _dirs[0] == "~":
        root_dir = Path.home()
        path = root_dir / ("/".join(_dirs[1:]))
    else:
        root_dir = Path.cwd()
        path = root_dir / ("/".join(_dirs[1:]))

    path.mkdir(exist_ok=True, parents=True)

    return path
