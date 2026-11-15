import unittest
from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


class _ResetLoader(yaml.SafeLoader):
    pass


# Support Docker Compose's custom `!reset` tag when parsing deploy/docker/docker-compose.cpu.yml
_ResetLoader.add_constructor(
    "!reset",
    lambda loader, node: {"__reset__": loader.construct_mapping(node) if isinstance(node, yaml.MappingNode) else None},
)


class DockerCpuGpuConfigTests(unittest.TestCase):
    def test_docker_compose_supports_camera_device_variable(self):
        compose_path = REPO_ROOT / "docker-compose.yml"
        data = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
        services = data["services"]

        checkout = services["camera_checkout"]
        self.assertEqual(
            checkout["build"]["args"]["CAMERA_DEVICE"],
            "${CAMERA_DEVICE:-cuda}",
        )
        self.assertIn("${CAMERA_DEVICE:-cuda}", checkout["command"])

        entrance = services["camera_entrance"]
        self.assertEqual(
            entrance["build"]["args"]["CAMERA_DEVICE"],
            "${CAMERA_DEVICE:-cuda}",
        )
        self.assertIn("${CAMERA_DEVICE:-cuda}", entrance["command"])

    def test_docker_compose_cpu_override_strips_gpu_reservation_and_defaults_to_cpu(self):
        cpu_compose_path = REPO_ROOT / "deploy" / "docker" / "docker-compose.cpu.yml"
        self.assertTrue(cpu_compose_path.exists())
        data = yaml.load(cpu_compose_path.read_text(encoding="utf-8"), Loader=_ResetLoader)
        services = data["services"]

        for svc_name in ("camera_checkout", "camera_entrance"):
            svc = services[svc_name]
            self.assertEqual(svc["build"]["args"]["CAMERA_DEVICE"], "${CAMERA_DEVICE:-cpu}")
            self.assertIn("${CAMERA_DEVICE:-cpu}", svc["command"])
            self.assertIn("__reset__", svc["deploy"])

    def test_dockerfile_camera_has_cuda_and_cpu_base_stages(self):
        dockerfile = (REPO_ROOT / "deploy" / "docker" / "Dockerfile.camera").read_text(encoding="utf-8")
        self.assertIn("ARG CAMERA_DEVICE=cuda", dockerfile)
        self.assertIn("AS base-cuda", dockerfile)
        self.assertIn("FROM python:3.10.16-slim-bookworm AS base-cpu", dockerfile)
        self.assertIn("FROM base-${CAMERA_DEVICE}", dockerfile)
        self.assertIn("s/^onnxruntime-gpu==/onnxruntime==/", dockerfile)

    def test_env_example_documents_camera_device_and_cpu_compose_file(self):
        env_example = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
        self.assertIn("CAMERA_DEVICE=", env_example)
        self.assertIn("deploy/docker/docker-compose.cpu.yml", env_example)


if __name__ == "__main__":
    unittest.main()
