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
    def test_docker_compose_supports_camera_device_and_openvino_env(self):
        compose_path = REPO_ROOT / "docker-compose.yml"
        data = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
        services = data["services"]

        for svc_name in ("camera_checkout", "camera_entrance"):
            svc = services[svc_name]
            self.assertEqual(
                svc["build"]["args"]["CAMERA_DEVICE"],
                "${CAMERA_DEVICE:-cuda}",
            )
            self.assertIn("${CAMERA_DEVICE:-cuda}", svc["command"])
            self.assertEqual(svc["environment"]["OPENVINO_DEVICE"], "${OPENVINO_DEVICE:-CPU}")
            self.assertEqual(svc["environment"]["OPENVINO_PRECISION"], "${OPENVINO_PRECISION:-FP32}")

    def test_docker_compose_cpu_override_strips_gpu_reservation_and_defaults_to_openvino(self):
        cpu_compose_path = REPO_ROOT / "deploy" / "docker" / "docker-compose.cpu.yml"
        self.assertTrue(cpu_compose_path.exists())
        data = yaml.load(cpu_compose_path.read_text(encoding="utf-8"), Loader=_ResetLoader)
        services = data["services"]

        for svc_name in ("camera_checkout", "camera_entrance"):
            svc = services[svc_name]
            self.assertEqual(svc["build"]["args"]["CAMERA_DEVICE"], "${CAMERA_DEVICE:-openvino}")
            self.assertIn("${CAMERA_DEVICE:-openvino}", svc["command"])
            self.assertIn("__reset__", svc["deploy"])

    def test_dockerfile_camera_has_cuda_cpu_and_openvino_stages(self):
        dockerfile = (REPO_ROOT / "deploy" / "docker" / "Dockerfile.camera").read_text(encoding="utf-8")
        self.assertIn("ARG CAMERA_DEVICE=cuda", dockerfile)
        self.assertIn("AS base-cuda", dockerfile)
        self.assertIn("FROM python:3.10.16-slim-bookworm AS base-cpu", dockerfile)
        self.assertIn("FROM base-cpu AS base-openvino", dockerfile)
        self.assertIn("FROM base-${CAMERA_DEVICE}", dockerfile)
        self.assertIn("onnxruntime-openvino==1.23.0", dockerfile)
        self.assertIn("openvino==2025.3.0", dockerfile)

    def test_env_example_documents_camera_device_and_cpu_compose_file(self):
        env_example = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
        self.assertIn("CAMERA_DEVICE=", env_example)
        self.assertIn("OPENVINO_DEVICE=CPU", env_example)
        self.assertIn("deploy/docker/docker-compose.cpu.yml", env_example)


if __name__ == "__main__":
    unittest.main()
