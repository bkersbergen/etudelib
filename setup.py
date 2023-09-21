"""Setup file for etudelib."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import List

from setuptools import find_packages, setup


def load_module(name: str = "etudelib/__init__.py"):
    """Load Python Module.

    Args:
        name (str, optional): Name of the module to load.
            Defaults to "etudelib/__init__.py".

    Returns:
        _type_: _description_
    """
    location = str(Path(__file__).parent / name)
    spec = spec_from_file_location(name=name, location=location)
    module = module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(module)  # type: ignore
    return module


def get_version() -> str:
    """Get version from `etudelib.__init__`.

    Version is stored in the main __init__ module in `etudelib`.
    The varible storing the version is `__version__`. This function
    reads `__init__` file, checks `__version__ variable and return
    the value assigned to it.

    Example:
        >>> # Assume that __version__ = "0.0.1"
        >>> get_version()
        "0.0.1"

    Returns:
        str: `etudelib` version.
    """
    etude2 = load_module(name="etudelib/__init__.py")
    version = etude2.__version__
    return version


def get_required_packages(requirement_files: List[str]) -> List[str]:
    """Get packages from requirements.txt file.

    This function returns list of required packages from requirement files.

    Args:
        requirement_files (List[str]): txt files that contains list of required
            packages.

    Example:
        >>> get_required_packages(requirement_files=["onnx"])
        ['onnx>=1.8.1', 'networkx~=2.5', ...]

    Returns:
        List[str]: List of required packages
    """

    required_packages: List[str] = []

    for requirement_file in requirement_files:
        with open(f"requirements/{requirement_file}.txt", "r", encoding="utf8") as file:
            for line in file:
                package = line.strip()
                if package and not package.startswith(("#", "-f")):
                    required_packages.append(package)

    return required_packages


VERSION = get_version()
LONG_DESCRIPTION = (Path(__file__).parent / "README.md").read_text(encoding="utf8")
INSTALL_REQUIRES = get_required_packages(requirement_files=["base"])
EXTRAS_REQUIRE = {
    "dev": get_required_packages(requirement_files=["dev"]),
    "full": get_required_packages(requirement_files=["docs", "dev"]),
}


setup(
    name="etudelib",
    version=get_version(),
    author="bolcom",
    author_email="bkersbergen@bol.com",
    description="etude",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="",
    license="Copyright (c)"
    'Licensed under the Apache License, Version 2.0 (the "License")'
    "See LICENSE file for more details.",
    python_requires=">=3.8",
    packages=find_packages(exclude=("tests",)),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    package_data={"": ["config.yaml"]},
    entry_points={"console_scripts": ["etudelib=etudelib.utils.cli.cli:main"]},
)