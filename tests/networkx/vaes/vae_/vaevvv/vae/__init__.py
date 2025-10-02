"""
from . import base
from . import origin
from . import hvae
from . import priors
from . import timevae
"""

hard_dependencies = ("numpy","pandas","torch","layertools")
missing_dependencies = []


for dependency in hard_dependencies:
    try:
        if len(dependency.split("==")) == 2:
            module_name, module_version = dependency.split("==")
            module = __import__(module_name)
            if module.__version__ != module_version:
                missing_dependencies.append(f"{module_name}: the package need version {module_version},now version is mismatching")
        else:
            __import__(dependency)
    except ImportError as e:
        missing_dependencies.append(f"{dependency}: {e}")

if missing_dependencies:
    raise ImportError(
        "Unable to import required dependencies:\n" + "\n".join(missing_dependencies)
    )

del hard_dependencies, dependency, missing_dependencies

from .__version__ import get_versions
v = get_versions()
__version__ = v.get("closest-tag", v["version"])
del get_versions, v

try:
    from . import _test_sign
except:
    pass

from . import base
from . import origin
# from . import hvae
from . import gtmprior
from . import vamprior
from . import timevae


