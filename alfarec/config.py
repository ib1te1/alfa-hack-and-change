from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError


class ConfigurationError(ValueError):
    pass


class UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        self.flatten_mapping(node)
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in result:
                raise yaml.constructor.ConstructorError(
                    "while reading a mapping",
                    node.start_mark,
                    f"duplicate key: {key}",
                    key_node.start_mark,
                )
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def load_configuration[Model: BaseModel](path: Path, model: type[Model]) -> Model:
    try:
        payload = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
        return model.model_validate(payload)
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError, TypeError) as error:
        raise ConfigurationError(f"Invalid configuration at {path}: {error}") from error
