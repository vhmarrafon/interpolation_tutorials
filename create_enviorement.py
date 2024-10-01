import os
import sys
import subprocess
import platform
import argparse
import logging

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_venv(venv_name):
    """
    create the virtual environment using python venv

    :param venv_name:   str   - venv name string
    """
    logging.info(f"Creating virtual environment: {venv_name}")
    subprocess.check_call([sys.executable, "-m", "venv", venv_name])


def activate_venv(venv_name):
    """
    active the virtual environment, venv

    :param venv_name:   str   - venv name string
    """
    if platform.system() == "Windows":
        activate_script = os.path.join(venv_name, "Scripts", "activate")
    else:
        activate_script = os.path.join(venv_name, "bin", "activate")

    return activate_script


def create_conda_env(env_name):
    """
    create the virtual environment using conda

    :param venv_name:   str   - venv name string
    """
    logging.info(f"Creating Conda environment: {env_name}")
    subprocess.check_call(["conda", "create", "-n", env_name, "python=3.8", "-y"])


def activate_conda_env(env_name):
    """
    active the virtual environment, conda

    :param venv_name:   str   - venv name string
    """
    subprocess.call(f"conda activate {env_name}", shell=True)


def install_requirements(env_type, venv_name):
    """
    install packages from requirements.txt and enable venv as jupyter kernel

    :param venv_name:   str   - venv name string
    :param env_type:    str   - env type, conda or venv
    """
    logging.info("Installing dependencies from requirements.txt...")
    if env_type == "venv":
        activate_cmd = activate_venv(venv_name)
        subprocess.call(f"{activate_cmd} && pip install -r requirements.txt", shell=True)
        install_jupyter(env_type, venv_name)
    elif env_type == "conda":
        subprocess.call("pip install -r requirements.txt", shell=True)
        install_jupyter(env_type, venv_name)


def install_jupyter(env_type, venv_name):
    """
    install jupyter kernel

    :param venv_name:   str   - venv name string
    :param env_type:    str   - env type, conda or venv
    """
    logging.info("Installing Jupyter Notebook...")
    if env_type == "venv":
        activate_cmd = activate_venv(venv_name)
        subprocess.call(f"{activate_cmd} && pip install jupyter", shell=True)
        logging.info("Configuring Jupyter kernel for venv...")
        subprocess.check_call(f"{activate_cmd} && python -m ipykernel install --user --name={venv_name}", shell=True)
    elif env_type == "conda":
        subprocess.call("pip install jupyter", shell=True)
        logging.info("Configuring Jupyter kernel for conda...")
        subprocess.call(f"python -m ipykernel install --user --name={venv_name}", shell=True)


def setup_environment(env_type, env_name):
    """
    function to setup the env

    :param venv_name:   str   - venv name string
    :param env_type:    str   - env type, conda or venv
    """

    if env_type == "venv":
        create_venv(env_name)
        logging.info(f"Virtual environment '{env_name}' created.")
        install_requirements(env_type, env_name)

    elif env_type == "conda":
        create_conda_env(env_name)
        logging.info(f"Conda environment '{env_name}' created.")
        activate_conda_env(env_name)
        install_requirements(env_type, env_name)


def main():

    parser = argparse.ArgumentParser(description="create a environment using conda or venv.")

    parser.add_argument('--env-type', choices=['venv', 'conda'], required=True,
                        help="choose the environment type ('venv' or 'conda')")
    parser.add_argument('--env-name', required=True,
                        help="set the environment name.")

    args = parser.parse_args()

    # call process
    setup_environment(args.env_type, args.env_name)


if __name__ == "__main__":
    main()
