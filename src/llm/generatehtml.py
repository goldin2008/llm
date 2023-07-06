import os
from pathlib import Path

import pdoc

modules = ["test", "training"]  # Public submodules are auto-imported
context = pdoc.Context()
modules = [pdoc.Module(mod, context=context) for mod in modules]
pdoc.link_inheritance(context)


def recursive_htmls(mod):
    """
    Function to get htmls

    Returns:
        html
    """
    yield mod.name, mod.html()
    for submod in mod.submodules():
        yield from recursive_htmls(submod)


for mod in modules:
    for module_name, html in recursive_htmls(mod):
        print(module_name)  # Process
        # print(html)
        # parent_folder_path = os.path.abspath(
        #     os.path.join(os.path.dirname(__file__), "html")
        # )
        path = Path(os.path.dirname(__file__))
        html_path = os.path.join(path.parent.parent, "html")
        html_file = html_path + "/" + module_name + ".html"
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html)
