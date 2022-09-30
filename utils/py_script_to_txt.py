
def save_config_file(config_python_script: str, config_txt_path: str) -> None:
    """For reproductibility, it is good to save the config file.

    Args:
        config_python_script (str): config python script
        config_txt_path (str): path to saved config file
    """
    with open(config_python_script) as f:
        data = f.read()
        f.close()

    with open(config_txt_path, mode="w") as f:
        f.write(data)
        f.close()

