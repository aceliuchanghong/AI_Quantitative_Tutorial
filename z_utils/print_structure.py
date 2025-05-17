from tree_utils.struct_tree_out import print_tree

# import os
# import sys

# sys.path.insert(
#     0,
#     os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")),
# )

path = "../python_template"

exclude_dirs_set = {
    "z_utils",
    ".env",
    "LICENSE",
    ".gitignore",
    ".python-version",
    "pyproject.toml",
    "README.md",
    "uv.lock",
    "prompt.md",
    "package-lock.json",
    "node_modules",
    "z_using_files",
}
print_tree(directory=path, exclude_dirs=exclude_dirs_set)
# uv run z_utils/print_structure.py
