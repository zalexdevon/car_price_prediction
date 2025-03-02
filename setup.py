import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.0"

REPO_NAME = "car_price_prediction"  # tên của github repo
AUTHOR_USER_NAME = "trantamhocjava"  # tên của tài khoản github
SRC_REPO = "regressor"  # tên thư mục chứa source code, lưu ý là đang phân loại hay hồi quy
AUTHOR_EMAIL = "trantamhocjava@gmail.com"  # email đăng kí github

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="a small python package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
)
