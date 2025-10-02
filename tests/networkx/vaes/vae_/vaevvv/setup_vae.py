from distutils.core import setup
from vae.__version__ import get_versions

v = get_versions()

setup(name='vae',  # 包名
      version=v['version'],  # 版本号
      author='zhaoxuefeng',  # 作者
      packages=[
                'vae.gtmprior',
                'vae.origin',
                'vae.timevae',
                # 'vae.signature_tf',
                # 'vae.signature_torch',
                'vae.vamprior',
                ],  # 包列表
      py_modules=['vae.__version__','vae.__init__','vae.base'],
      description='vae 方面的常用函数聚合',
      )


