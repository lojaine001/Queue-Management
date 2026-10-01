"""Immutable model generations with an atomic current-generation pointer."""
from pathlib import Path
import shutil
import uuid
from prediction.runtime import read_json, write_json


def current_bundle(models):
    models = Path(models)
    version = read_json(models / 'current.json').get('version')
    if version and Path(version).name == version:
        path = models / 'versions' / version
        if path.is_dir():
            return path
    return models


class ModelBundle:
    def __init__(self, models, names, legacy_dirs=()):
        self.models = Path(models)
        self.names = list(names)
        self.previous = current_bundle(models)
        self.version = uuid.uuid4().hex
        self.stage = self.models / 'versions' / self.version
        self.stage.mkdir(parents=True)
        for name in self.names:
            for directory in [self.previous, *map(Path, legacy_dirs)]:
                source = directory / name
                if source.exists():
                    shutil.copy2(source, self.stage / name)
                    break
        self.original = self.signature()
        self.promoted = False

    def signature(self):
        return {name: (p.stat().st_size, p.stat().st_mtime_ns) for name in self.names
                if (p := self.stage / name).exists()}

    def changed(self):
        return self.original != self.signature()

    def promote(self, required, metadata):
        missing = [name for name in required if not (self.stage / name).is_file()]
        if missing:
            raise RuntimeError(f'Incomplete model bundle: {missing}')
        write_json(self.stage / 'bundle.json', metadata)
        write_json(self.models / 'current.json', {'version': self.version, **metadata})
        self.promoted = True
        return self.version

    def cleanup(self):
        if not self.promoted:
            shutil.rmtree(self.stage, ignore_errors=True)
