from pathlib import Path
from setuptools import setup
import re

_pwd = Path(__file__).parent.absolute()


def main():
    cmdclass = dict()

    version = None
    init_path = _pwd / "lidarnerf" / "__init__.py"
    with open(init_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            match_res = re.match(r'^__version__ = "(.*)"', line)
            if match_res:
                version = match_res.group(1)
                break
    if version is None:
        raise RuntimeError(f"Cannot find version from {init_path}")
    print(f"Detected lidarnerf version: {version}")

    _ = setup(
        name="lidarnerf",
        version=version,
        description="LiDAR-NeRF: Novel LiDAR View Synthesis via Neural Radiance Fields",
        packages=["lidarnerf", "lidarnvs"],
        cmdclass=cmdclass,
        include_package_data=True,
    )


if __name__ == "__main__":
    main()
