from setuptools import setup, find_packages

setup(
    name="jims_ml_sandbox",
    version="0.1.0",
    packages=find_packages(),  # Automatically discover all packages and subpackages
    python_requires='>=3.10',
    install_requires=[
        # List your dependencies here (e.g., numpy, pandas, etc.)
    ],
    author="Jim Coles",  # Replace with your name or organization
    author_email="jim@prospace.com",  # Replace with your email
    description="Jim's ML Sandbox",
    url="https://github/jimcoles/jim_ml_sandbox",  # Project homepage or repository URL
)