import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FISHEYE_DIR = REPO_ROOT / "Fisheye-Detector"


def _load_fisheye_yolo():
    spec = importlib.util.spec_from_file_location(
        "fisheye_yolo", FISHEYE_DIR / "utils" / "yolo.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fisheye_yolo"] = mod
    spec.loader.exec_module(mod)
    return mod


class DummyOnnxSession:
    """Mock ONNX Runtime InferenceSession for testing all YOLO output layouts."""

    def __init__(self, input_shape, output_tensors, output_names=None):
        self._input_shape = input_shape
        self._output_tensors = output_tensors
        self._output_names = output_names or [f"out_{i}" for i in range(len(output_tensors))]

    def get_inputs(self):
        return [SimpleNamespace(name="images", shape=self._input_shape, type="tensor(float)")]

    def get_outputs(self):
        return [
            SimpleNamespace(name=name, shape=list(arr.shape))
            for name, arr in zip(self._output_names, self._output_tensors)
        ]

    def run(self, output_names, input_feed):
        return self._output_tensors


class TestFisheyeDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.yolo_mod = _load_fisheye_yolo()

    def _make_stub_detector(self, session, model_path="yolov13l_fisheye8k.onnx", person_class_ids=None):
        det = object.__new__(self.yolo_mod.FisheyeDetector)
        det._model_path = model_path
        det._obj_class_score_th = 0.25
        det._attr_class_score_th = 0.70
        det._iou_th = 0.45
        det._default_img_size = 640
        det._explicit_person_class_ids = set(person_class_ids) if person_class_ids is not None else None
        det._backend = "onnx"
        det._interpreter = session
        det._model = session.run
        det._input_shapes = [inp.shape for inp in session.get_inputs()]
        det._input_names = [inp.name for inp in session.get_inputs()]
        det._input_dtypes = [np.float32 for _ in session.get_inputs()]
        det._output_shapes = [out.shape for out in session.get_outputs()]
        det._output_names = [out.name for out in session.get_outputs()]
        det._pinto_style_graph = (
            det._output_names == ["x1y1x2y2", "main01_scores"]
        )
        return det

    def test_yolov9_alias_and_box_api(self):
        """FisheyeDetector exports YOLOv9 alias and Box dataclass matching Head-Detector."""
        self.assertIs(self.yolo_mod.YOLOv9, self.yolo_mod.FisheyeDetector)
        box = self.yolo_mod.Box(
            classid=0, score=0.88, x1=10, y1=20, x2=110, y2=220, cx=60, cy=120
        )
        self.assertEqual(box.classid, 0)
        self.assertAlmostEqual(box.score, 0.88)
        self.assertEqual((box.x1, box.y1, box.x2, box.y2), (10, 20, 110, 220))

    def test_letterbox_and_unscale_roundtrip(self):
        """Letterbox image and unscale_letterbox_coords preserve original image coordinates."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        lb, ratio, (dw, dh) = self.yolo_mod.letterbox_image(img, (640, 640))
        self.assertEqual(lb.shape, (640, 640, 3))
        self.assertAlmostEqual(ratio[0], 1.0)
        self.assertAlmostEqual(ratio[1], 1.0)
        self.assertAlmostEqual(dw, 0.0)
        self.assertAlmostEqual(dh, 80.0)

        # A box at (100, 50, 200, 150) in original 640x480 sits at y=(130..230) in 640x640 letterbox
        boxes_lb = np.array([[100.0, 130.0, 200.0, 230.0]], dtype=np.float32)
        unscaled = self.yolo_mod.unscale_letterbox_coords(boxes_lb, (480, 640), ratio, (dw, dh))
        np.testing.assert_allclose(unscaled[0], [100.0, 50.0, 200.0, 150.0], atol=1e-4)

    def test_yolov10_end2end_fisheye8k_pedestrian_normalized_to_class_0(self):
        """YOLOv10 [1, 300, 6] output keeps Fisheye8K Pedestrian (class 3) and normalizes to classid=0."""
        out = np.zeros((1, 10, 6), dtype=np.float32)
        # Detection 0: Pedestrian (class 3) in Fisheye8K
        out[0, 0] = [100.0, 130.0, 200.0, 250.0, 0.85, 3.0]
        # Detection 1: Car (class 2) in Fisheye8K -> normalized to classid=2 (not 0)
        out[0, 1] = [300.0, 200.0, 400.0, 350.0, 0.92, 2.0]

        sess = DummyOnnxSession([1, 3, 640, 640], [out])
        det = self._make_stub_detector(sess, person_class_ids=[3])
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        boxes = det(image=img, disable_headpose_identification_mode=True)
        person_boxes = [b for b in boxes if b.classid == 0]

        self.assertEqual(len(person_boxes), 1)
        self.assertEqual(person_boxes[0].classid, 0)
        self.assertAlmostEqual(person_boxes[0].score, 0.85, places=4)
        self.assertEqual((person_boxes[0].x1, person_boxes[0].y1, person_boxes[0].x2, person_boxes[0].y2), (100, 50, 200, 170))

    def test_yolov13_transposed_output_decoding(self):
        """YOLOv13/v8/v9 [1, 4+C, A] output auto-detects 5-class Fisheye8K Pedestrian (class 3 -> 0)."""
        # 5 classes (Bus, Bike, Car, Pedestrian, Truck) => shape [1, 9, 20]
        out = np.zeros((1, 9, 20), dtype=np.float32)
        # Anchor 0: cx=150, cy=180, w=60, h=80 in 640x640 letterbox, class 3 (Pedestrian) score=0.80
        out[0, 0, 0] = 150.0
        out[0, 1, 0] = 180.0
        out[0, 2, 0] = 60.0
        out[0, 3, 0] = 80.0
        out[0, 4 + 3, 0] = 0.80
        # Anchor 1: Car (class 2) score=0.90 -> classid=2 (not 0)
        out[0, 0, 1] = 400.0
        out[0, 1, 1] = 300.0
        out[0, 2, 1] = 100.0
        out[0, 3, 1] = 100.0
        out[0, 4 + 2, 1] = 0.90

        sess = DummyOnnxSession([1, 3, 640, 640], [out])
        det = self._make_stub_detector(sess, person_class_ids=None)  # auto-detects {3} for nc=5
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        boxes = det(image=img)
        person_boxes = [b for b in boxes if b.classid == 0]

        self.assertEqual(len(person_boxes), 1)
        self.assertEqual(person_boxes[0].classid, 0)
        self.assertAlmostEqual(person_boxes[0].score, 0.80, places=4)
        self.assertEqual((person_boxes[0].x1, person_boxes[0].y1, person_boxes[0].x2, person_boxes[0].y2), (120, 60, 180, 140))

    def test_yolor_anchor_output_decoding(self):
        """YOLOR [1, A, 5+C] output multiplies obj_conf * cls_conf and normalizes class 3 -> 0."""
        out = np.zeros((1, 20, 10), dtype=np.float32)
        # Anchor 0: cx=320, cy=320, w=80, h=100, obj=0.9, class 3 (Pedestrian)=0.8 -> score=0.72
        out[0, 0, :4] = [320.0, 320.0, 80.0, 100.0]
        out[0, 0, 4] = 0.90
        out[0, 0, 5 + 3] = 0.80

        sess = DummyOnnxSession([1, 3, 640, 640], [out])
        det = self._make_stub_detector(sess, person_class_ids=None)
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        boxes = det(image=img)

        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0].classid, 0)
        self.assertAlmostEqual(boxes[0].score, 0.72, places=4)
        self.assertEqual((boxes[0].x1, boxes[0].y1, boxes[0].x2, boxes[0].y2), (280, 270, 360, 370))

    def test_weighted_boxes_fusion_np(self):
        """WBF fuses overlapping detections from YOLOR, YOLOv10, and YOLOv13."""
        boxes_list = [
            np.array([[0.10, 0.10, 0.30, 0.30]], dtype=np.float32),
            np.array([[0.11, 0.10, 0.31, 0.30]], dtype=np.float32),
            np.array([[0.10, 0.11, 0.30, 0.31]], dtype=np.float32),
        ]
        scores_list = [
            np.array([0.80], dtype=np.float32),
            np.array([0.90], dtype=np.float32),
            np.array([0.85], dtype=np.float32),
        ]
        labels_list = [
            np.array([0], dtype=np.int64),
            np.array([0], dtype=np.int64),
            np.array([0], dtype=np.int64),
        ]
        fused_b, fused_s, fused_l = self.yolo_mod.weighted_boxes_fusion_np(
            boxes_list, scores_list, labels_list, weights=[9.0, 9.0, 9.0], iou_thr=0.65, skip_box_thr=0.01
        )
        self.assertEqual(len(fused_b), 1)
        self.assertEqual(int(fused_l[0]), 0)
        self.assertAlmostEqual(float(fused_s[0]), (0.80 + 0.90 + 0.85) / 3.0, places=4)

    def test_rapid_rotated_nms_and_aabb_conversion(self):
        """RAPiD rotated NMS suppresses duplicate rotated boxes and converts to axis-aligned Box."""
        dets = np.array(
            [
                [200.0, 200.0, 60.0, 80.0, 0.0, 0.90],
                [202.0, 201.0, 60.0, 80.0, 0.0, 0.75],  # overlapping duplicate
                [500.0, 500.0, 50.0, 50.0, 0.0, 0.82],  # separate person
            ],
            dtype=np.float32,
        )
        kept = self.yolo_mod.rotated_nms(dets, thres=0.45)
        self.assertEqual(len(kept), 2)
        x1, y1, x2, y2 = self.yolo_mod.FisheyeDetector._rot_to_aabb(
            kept[0, 0], kept[0, 1], kept[0, 2], kept[0, 3], kept[0, 4], 1024, 1024
        )
        self.assertEqual((x1, y1, x2, y2), (170, 160, 230, 240))

    def test_rapid_onnx_output_decoding(self):
        """RAPiD ONNX [1, A, 6] output runs rotated NMS and returns Box(classid=0, rbox=...)."""
        out = np.zeros((1, 1500, 6), dtype=np.float32)
        # Anchor 0: cx=320, cy=320, w=60, h=80, angle=0 deg, score=0.88
        out[0, 0] = [320.0, 320.0, 60.0, 80.0, 0.0, 0.88]
        # Anchor 1: duplicate overlapping anchor at cx=322, cy=321, score=0.70 -> suppressed by rotated NMS
        out[0, 1] = [322.0, 321.0, 60.0, 80.0, 0.0, 0.70]

        sess = DummyOnnxSession([1, 3, 640, 640], [out])
        det = self._make_stub_detector(sess, model_path="rapid_pL1_MWHB1024.onnx")
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        boxes = det(image=img)

        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0].classid, 0)
        self.assertAlmostEqual(boxes[0].score, 0.88, places=4)
        self.assertEqual((boxes[0].x1, boxes[0].y1, boxes[0].x2, boxes[0].y2), (290, 280, 350, 360))
        self.assertIsNotNone(boxes[0].rbox)


if __name__ == "__main__":
    unittest.main()
