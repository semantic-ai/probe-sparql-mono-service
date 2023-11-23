import fire

import json
import yaml


def main(
        input_file: str = "example_config.json",
        output_file: str = "yaml_config.yml",
):
    content = json.load(open(input_file, "r"))
    yaml.safe_dump(content, open(output_file, "w"))


if __name__ == "__main__":
    fire.Fire(main)
