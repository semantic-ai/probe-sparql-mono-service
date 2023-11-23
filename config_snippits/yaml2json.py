import fire

import json
import yaml


def main(
        input_path: str = "yaml_config.yml",
        output_path: str = "example_config.json"
):
    content = yaml.safe_load(open(input_path, "r"))
    json.dump(content, open(output_path, "w+"))


if __name__ == "__main__":
    fire.Fire(main)
